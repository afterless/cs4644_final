import argparse
import copy
import os
import sys

import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import wandb
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mlp_grok import MLP
from train.mlp_grok_train import cross_entropy_high_precision, gen_train_test
from utils.weight_matching import (
    weight_matching,
    apply_permutation,
    mlp_grok_permutation_spec,
)
from utils.activation_matching import activation_matching
from utils.straight_through_estimator import straight_through_estimator
from utils.util import lerp
from utils.training import test
from utils.plot import plot_interp_acc

# Fixed params for testing
p = 113
d_model = 64
d_vocab = p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=int, default=1)
    parser.add_argument("--model_b", type=int, default=3)
    parser.add_argument(
        "--matching", type=str, default="wm", choices=["wm", "ste", "act"]
    )
    parser.add_argument("--stopping_thresh", type=float, default=3e-6)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    chkptA = t.load(
        f"../train/checkpoints/mlp_grok/mlp_grok_final_{args.model_a}.pt",
        map_location=device,
    )
    chkptB = t.load(
        f"../train/checkpoints/mlp_grok/mlp_grok_final_{args.model_b}.pt",
        map_location=device,
    )

    modelA = MLP(d_vocab, d_model)
    modelA.to(device)
    modelA.load_state_dict(chkptA)

    modelB = MLP(d_vocab, d_model)
    modelB.to(device)
    modelB.load_state_dict(chkptB)

    ps = mlp_grok_permutation_spec(num_hidden_layers=2)
    train_pairs, test_pairs = gen_train_test(0.3, p, seed=1)

    train_pairs = TensorDataset(
        t.tensor([t[0] for t in train_pairs], dtype=t.long),
        t.tensor([t[1] for t in train_pairs], dtype=t.long),
    )

    test_pairs = TensorDataset(
        t.tensor([t[0] for t in test_pairs], dtype=t.long),
        t.tensor([t[1] for t in test_pairs], dtype=t.long),
    )

    train_loader = DataLoader(train_pairs, batch_size=len(train_pairs), num_workers=2)

    test_loader = DataLoader(test_pairs, batch_size=len(test_pairs), num_workers=2)

    if args.matching == "wm":
        opt_perm = weight_matching(
            ps,
            modelA.state_dict(),
            modelB.state_dict(),
        )
    elif args.matching == "act":
        opt_perm = activation_matching(
            ps,
            modelA,
            modelB,
            train_loader,
            device,
        )
    elif args.matching == "ste":
        wandb.init(project="perm_matching", config=vars(args))
        opt_perm = straight_through_estimator(
            ps,
            modelA,
            modelB,
            train_loader,
            test_loader,
            cross_entropy_high_precision,
            device,
            args,
        )
    else:
        raise ValueError(f"Unknown matching {args.matching}")

    updated_params = apply_permutation(ps, opt_perm, modelB.state_dict())

    lam = t.linspace(0, 1, steps=25)

    test_loss_interp_clever = []
    test_loss_interp_naive = []
    train_loss_interp_clever = []
    train_loss_interp_naive = []

    # naive
    modelB.load_state_dict(chkptB)
    modelA_dict = copy.deepcopy(modelA.state_dict())
    modelB_dict = copy.deepcopy(modelB.state_dict())
    i = 0
    for l in tqdm(lam):
        naive_p = lerp(l, modelA_dict, modelB_dict)
        modelB.load_state_dict(naive_p)
        test_loss, _ = test(modelB, device, i, test_loader, F.cross_entropy)
        test_loss_interp_naive.append(test_loss)
        train_loss, _ = test(modelB, device, i, train_loader, F.cross_entropy)
        train_loss_interp_naive.append(train_loss)
        i += 1

    # clever
    modelB.load_state_dict(updated_params)
    modelA_dict = copy.deepcopy(modelA.state_dict())
    modelB_dict = copy.deepcopy(modelB.state_dict())
    i = 0
    for l in tqdm(lam):
        clever_p = lerp(l, modelA_dict, modelB_dict)
        modelB.load_state_dict(clever_p)
        test_loss, _ = test(modelB, device, i, test_loader, F.cross_entropy)
        test_loss_interp_clever.append(test_loss)
        train_loss, _ = test(modelB, device, i, train_loader, F.cross_entropy)
        train_loss_interp_clever.append(train_loss)
        i += 1

    fig = plot_interp_acc(
        lam,
        train_loss_interp_naive,
        test_loss_interp_naive,
        train_loss_interp_clever,
        test_loss_interp_clever,
    )

    os.makedirs("./plots", exist_ok=True)
    plt.savefig(
        f"./plots/mlp_grok_interp_{args.model_a}_{args.model_b}_{args.matching}.png",
        dpi=300,
    )


if __name__ == "__main__":
    main()
