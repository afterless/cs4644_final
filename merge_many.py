"""Train many MNIST MLPs, each seeing a random subset of the dataset. Then merge
the models with MergeMany and evaluate calibration, etc."""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import jit, random, tree_map, value_and_grad, vmap
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from models.mlp_grok import MLP
from utils.training import test
from utils.weight_matching import PermutationSpec, mlp_permutation_spec, weight_matching, apply_permutation
from train.mlp_grok_train import gen_train_test, cross_entropy_high_precision

# See https://github.com/tensorflow/tensorflow/issues/53831.

# See https://github.com/google/jax/issues/9454.

p = 113

def generate_numbers():
  """Return the training and test datasets, unbatched."""
  # See https://www.tensorflow.org/datasets/overview#as_batched_tftensor_batch_size-1.
  key = random.PRNGKey(0)
  inputs = jax.random.randint(key, (1500, 2), 0, p)
  targets = jnp.sum(inputs, axis=1) % p
  train_inputs, test_inputs = inputs[:1000], inputs[1000:]
  train_targets, test_targets = targets[:1000], targets[1000:]

  train_ds = {"inputs": train_inputs, "labels": train_targets}
  test_ds = {"inputs": test_inputs, "labels": test_targets}
  return train_ds, test_ds

activation = nn.relu

class MLPModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(512)(x)
    x = activation(x)
    x = nn.Dense(512)(x)
    x = activation(x)
    x = nn.Dense(1)(x)
    print(x)
    return x

def make_stuff(model):

  @jit
  def batch_eval(params, inputs, targets):
    predictions = model.apply({"params": params}, inputs)
    loss = jnp.mean(optax.l2_loss(predictions, jnp.reshape(targets, (500, 1))))
    num_correct = jnp.sum(predictions == targets)
    return loss, {"predictions": predictions, "num_correct": num_correct}

  @jit
  def step(train_state, inputs, labels):
    (l, info), g = value_and_grad(batch_eval, has_aux=True)(train_state.params, inputs, labels)
    return train_state.apply_gradients(grads=g), {"batch_loss": l, **info}

  def dataset_loss_and_accuracy(params, dataset, batch_size: int):
    num_examples = dataset["inputs"].shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    batch_ix = jnp.arange(num_examples).reshape((num_batches, batch_size))
    # Can't use vmap or run in a single batch since that overloads GPU memory.
    losses, infos = zip(*[
        batch_eval(
            params,
            dataset["inputs"][batch_ix[i, :], :],
            dataset["labels"][batch_ix[i, :]],
        ) for i in range(num_batches)
    ])
    return (
        jnp.sum(batch_size * jnp.array(losses)) / num_examples,
        sum(x["num_correct"] for x in infos) / num_examples,
    )

  def dataset_logits(params, dataset, batch_size: int):
    num_examples = dataset["inputs"].shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    batch_ix = jnp.arange(num_examples).reshape((num_batches, batch_size))
    # Can't use vmap or run in a single batch since that overloads GPU memory.
    _, infos = zip(*[
        batch_eval(
            params,
            dataset["inputs"][batch_ix[i, :], :],
            dataset["labels"][batch_ix[i, :]],
        ) for i in range(num_batches)
    ])
    return jnp.concatenate([x["predictions"] for x in infos])

  return {
      "batch_eval": batch_eval,
      "step": step,
      "dataset_loss_and_accuracy": dataset_loss_and_accuracy,
      "dataset_logits": dataset_logits
  }

if __name__ == "__main__":
  p = 113
  d_model = 64
  d_vocab = p
  n_ctx = 3

  frac_train = 0.3
  n_models = 3
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  models = [MLP(d_vocab, n_ctx, d_model) for _ in range(n_models)]
  checkpoints = [torch.load(f"./train/checkpoints/mlp_grok_final_{i + 1}.pt", map_location=torch.device("cpu")) for i in range(n_models)]


  train_pairs, test_pairs = gen_train_test(frac_train, p)
  train_batch_size = len(train_pairs)
  test_batch_size = len(test_pairs)

  test_pairs = TensorDataset(
          torch.tensor([t[0] for t in test_pairs], dtype=torch.long),
          torch.tensor([t[1] for t in test_pairs], dtype=torch.long),
          )
  test_loader = DataLoader(
    test_pairs, batch_size=test_batch_size, shuffle=False, num_workers=2
  )
  losses = []
  epochs = []
  for i in range(n_models):
   models[i].load_state_dict(checkpoints[i])


  for i in range(n_models):
    model = models[i]
    epoch = None
    test_loss, test_acc = test(
        model, device, epoch, test_loader, cross_entropy_high_precision
    )
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

  ### Naive
  naive = MLP(d_vocab, n_ctx, d_model)
  with torch.no_grad():
      naive.embed.W_E.data = torch.mean(torch.stack([model.embed.W_E for model in models], dim=2))
      naive.layer0.weight.data = torch.mean(torch.stack([model.layer0.weight for model in models], dim=2))
      naive.layer1.weight.data = torch.mean(torch.stack([model.layer1.weight for model in models], dim=2))
      naive.unembed.W_U.data = torch.mean(torch.stack([model.unembed.W_U for model in models], dim=2))

  test_loss_naive, test_acc_naive = test(
      model, device, epoch, test_loader, cross_entropy_high_precision
  )
  print(f"test loss: {test_loss_naive:.4f}, test accuracy: {test_acc_naive:.4f}")

  permutation_spec = mlp_permutation_spec(2)

  def match2(p1, p2):
    perm = weight_matching(random.PRNGKey(123),
                           permutation_spec,
                           flatten_params(p1),
                           flatten_params(p2),
                           silent=True)
    p2_clever = unflatten_params(apply_permutation(permutation_spec, perm, flatten_params(p2)))
    return lerp(0.5, p1, p2_clever)

  params01 = match2(all_params[0], all_params[1])
  test_loss_01, test_acc_01 = stuff['dataset_loss_and_accuracy'](params01, test_ds, 500)
  print(f"test loss 0->1: {test_loss_01:.4f}, test accuracy 0->1: {test_acc_01:.4f}")

  def match_many(rng, permutation_spec: PermutationSpec, ps, max_iter=100):
    for iteration in range(max_iter):
      progress = False
      for p_ix in random.permutation(rngmix(rng, iteration), len(ps)):
        other_models_mean = tree_mean(ps[:p_ix] + ps[p_ix + 1:])
        l2_before = tree_l2(other_models_mean, ps[p_ix])
        perm = weight_matching(rngmix(rng, f"{iteration}-{p_ix}"),
                               permutation_spec,
                               flatten_params(other_models_mean),
                               flatten_params(ps[p_ix]),
                               silent=True)
        ps[p_ix] = unflatten_params(
            apply_permutation(permutation_spec, perm, flatten_params(ps[p_ix])))
        l2_after = tree_l2(other_models_mean, ps[p_ix])
        progress = progress or l2_after < l2_before - 1e-12
        print(f"iteration {iteration}/model {p_ix}: l2 diff {l2_after - l2_before:.4f}")

      if not progress:
        break

    return ps

  params_barycenter = tree_mean(match_many(random.PRNGKey(123), permutation_spec, all_params))
  train_loss_barycenter, train_acc_barycenter = stuff["dataset_loss_and_accuracy"](
      params_barycenter, train_ds, 500)
  test_loss_barycenter, test_acc_barycenter = stuff["dataset_loss_and_accuracy"](params_barycenter,
                                                                                 test_ds, 500)
  print(
      f"[barycenter] train loss: {train_loss_barycenter:.4f}, train accuracy: {train_acc_barycenter:.4f} "
      f"test loss: {test_loss_barycenter:.4f}, test accuracy: {test_acc_barycenter:.4f}")

  ### Plotting
  plt.figure(figsize=(12, 6))

  num_bins = 10
  bins = np.linspace(0, 1, num_bins + 1)
  bin_locations = 0.5 * (bins[:-1] + bins[1:])

  def one(bin_ix, probs, labels):
    lo, hi = bins[bin_ix], bins[bin_ix + 1]
    mask = (lo <= probs) & (probs <= hi)
    return np.mean(labels[jnp.reshape(mask, labels.shape)])

  # Train
  plt.subplot(1, 2, 1)
  plotting_ds = train_ds

  individual_model_logits = [
      stuff["dataset_logits"](p, plotting_ds, 500) for p in tqdm(all_params)
  ]
  barycenter_logits = stuff["dataset_logits"](params_barycenter, plotting_ds, 500)
  naive_logits = stuff["dataset_logits"](params_naive, plotting_ds, 500)
  ensemble_logits = sum(individual_model_logits) / num_models

  individual_model_probs = [jax.nn.softmax(x) for x in individual_model_logits]
  barycenter_probs = jax.nn.softmax(barycenter_logits)
  naive_probs = jax.nn.softmax(naive_logits)
  ensemble_probs = jax.nn.softmax(ensemble_logits)

  individual_model_ys = [[one(ix, probs, plotting_ds["labels"]) for ix in range(num_bins)]
                         for probs in tqdm(individual_model_probs)]
  wm_ys = [one(ix, barycenter_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]
  naive_ys = [one(ix, naive_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]
  ensemble_ys = [one(ix, ensemble_probs, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]

  plt.plot([0, 1], [0, 1], color="tab:grey", linestyle="dotted", label="Perfect calibration")

  plt.plot([], [], color="tab:orange", alpha=0.5, label="Individual models")
  for ys in individual_model_ys:
    plt.plot(bin_locations, ys, color="tab:orange", alpha=0.25)

  plt.plot(bin_locations,
           np.nan_to_num(naive_ys),
           color="tab:grey",
           marker=".",
           label="Naïve merging")
  plt.plot(bin_locations,
           np.nan_to_num(ensemble_ys),
           color="tab:purple",
           marker="2",
           label="Model ensemble")
  plt.plot(bin_locations, wm_ys, color="tab:green", marker="^", linewidth=2, label="MergeMany")
  plt.xlabel("Predicted probability")
  plt.ylabel("True probability")
  plt.axis("equal")
  plt.legend()
  plt.title("Train")
  plt.xlim(0, 1)
  plt.ylim(0, 1)
  plt.xticks(np.linspace(0, 1, 5))
  plt.yticks(np.linspace(0, 1, 5))

  # Test
  plt.subplot(1, 2, 2)
  plotting_ds = test_ds

  individual_model_logits = [
      stuff["dataset_logits"](p, plotting_ds, 500) for p in tqdm(all_params)
  ]
  barycenter_logits = stuff["dataset_logits"](params_barycenter, plotting_ds, 500)
  naive_logits = stuff["dataset_logits"](params_naive, plotting_ds, 500)
  ensemble_logits = sum(individual_model_logits) / num_models

  individual_model_preds = [x for x in individual_model_logits]
  barycenter_preds = barycenter_logits
  naive_preds = naive_logits
  ensemble_preds = ensemble_logits

  individual_model_ys = [[one(ix, probs, plotting_ds["labels"]) for ix in range(num_bins)]
                         for probs in tqdm(individual_model_preds)]
  wm_ys = [one(ix, barycenter_preds, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]
  naive_ys = [one(ix, naive_preds, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]
  ensemble_ys = [one(ix, ensemble_preds, plotting_ds["labels"]) for ix in tqdm(range(num_bins))]

  plt.plot([0, 1], [0, 1], color="tab:grey", linestyle="dotted", label="Perfect calibration")

  plt.plot([], [], color="tab:orange", alpha=0.5, linestyle="dashed", label="Individual models")
  for ys in individual_model_ys:
    plt.plot(bin_locations, ys, color="tab:orange", linestyle="dashed", alpha=0.25)

  plt.plot(bin_locations,
           np.nan_to_num(naive_ys),
           color="tab:grey",
           marker=".",
           linestyle="dashed",
           label="Naïve merging")
  plt.plot(bin_locations,
           np.nan_to_num(ensemble_ys),
           color="tab:purple",
           marker="2",
           linestyle="dashed",
           label="Model ensemble")
  plt.plot(bin_locations,
           wm_ys,
           color="tab:green",
           marker="^",
           linewidth=2,
           linestyle="dashed",
           label="MergeMany")
  plt.xlabel("Predicted probability")
  plt.ylabel("True probability")
  plt.axis("equal")
  # plt.legend()
  plt.title("Test")
  plt.xlim(0, 1)
  plt.ylim(0, 1)
  plt.xticks(np.linspace(0, 1, 5))
  plt.yticks(np.linspace(0, 1, 5))

  plt.suptitle("MergeMany Calibration")
  plt.tight_layout()
  plt.savefig("figs/mnist_wm_many_calibration_plot.png", dpi=300)
  plt.savefig("figs/mnist_wm_many_calibration_plot.pdf")
