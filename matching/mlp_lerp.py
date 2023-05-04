import argparse
import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import wandb
from tqdm import tqdm

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mlp_grok import MLP
from train.mlp_grok_train import cross_entropy_high_precision, gen_train_test
from utils.weight_matching import weight_matching, apply_permutation, mlp_grok_permutation_spec
from utils.straight_through_estimator import straight_through_estimator

# Fixed params for testing
p = 113
d_model = 64
d_vocab = p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frac_train", type=float, default=0.3)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1.0)

    args = parser.parse_args()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    chkptA = t.load("../train/checkpoints/mlp_grok_final_1.pt", map_location=device)
    chkptB = t.load("./checkpoints/perm_model_1_p3.pt", map_location=device)

    modelA = MLP(d_vocab, d_model)
    modelA.to(device)
    modelA.load_state_dict(chkptA)

    modelB = MLP(d_vocab, d_model)
    modelB.to(device)
    modelB.load_state_dict(chkptB)

    train_pairs, test_pairs = gen_train_test(args.frac_train, p, seed=args.seed)

    train_pairs = TensorDataset(
        t.tensor([t[0] for t in train_pairs], dtype=t.long),
        t.tensor([t[1] for t in train_pairs], dtype=t.long),
    )

    test_pairs = TensorDataset(
        t.tensor([t[0] for t in test_pairs], dtype=t.long),
        t.tensor([t[1] for t in test_pairs], dtype=t.long),
    )

    train_loader = DataLoader(
        train_pairs, batch_size=len(train_pairs), shuffle=True, num_workers=2
    )

    test_loader = DataLoader(
        test_pairs, batch_size=len(test_pairs), shuffle=True, num_workers=2
    )

    wandb.init(project="perm_matching", config=args)

    modelC = MLP(d_vocab, d_model)
    for p in t.arange(0, 1, step=0.1):
        modelC_dict = {}
        for (n, _), p_a, p_b in zip(modelC.named_parameters(), modelA.parameters(), modelB.parameters()):
            modelC_dict[n] = p * p_a + (1 - p) * p_b
        modelC.load_state_dict(modelC_dict)

        test_loss = 0
        i = 0
        correct = 0
        with t.inference_mode():
            for data, target in tqdm(test_loader):
                data, target = data.to(device), target.to(device)
                output = modelC(data)
                test_loss += cross_entropy_high_precision(output, target).item()
                pred = output.argmax(dim=-1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                i += 1
        test_loss /= i
        acc = 100.0 * correct / len(test_loader.dataset)
        print(
            "Average Test Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(test_loader.dataset), acc
            )
        )
        if wandb.run is not None:
            wandb.log({"test_loss": test_loss, "test_acc": acc, "lerp": p})
        