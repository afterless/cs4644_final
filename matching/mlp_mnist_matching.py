import argparse
import copy
import os
import sys

import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm
import wandb
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mlp_mnist import MLP
from utils.weight_matching import (
    weight_matching,
    apply_permutation,
    mlp_permutation_spec,
)
from utils.straight_through_estimator import straight_through_estimator
from utils.util import lerp
from utils.training import test
from utils.plot import plot_interp_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=int, default=1)
    parser.add_argument("--model_b", type=int, default=3)
    parser.add_argument(
        "--matching", type=str, default="wm", choices=["wm", "ste", "act"]
    )
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    chkptA = t.load(
        f"../train/checkpoints/mlp_mnist/mlp_mnist_final_{args.model_a}.pt",
        map_location=device,
    )
    chkptB = t.load(
        f"../train/checkpoints/mlp_mnist/mlp_mnist_final_{args.model_b}.pt",
        map_location=device,
    )

    modelA = MLP()
    modelA.to(device)
    modelA.load_state_dict(chkptA)

    modelB = MLP()
    modelB.to(device)
    modelB.load_state_dict(chkptB)

    ps = mlp_permutation_spec(num_hidden_layers=4)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    mnist_train = torchvision.datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    mnist_test = torchvision.datasets.MNIST(
        "../data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(mnist_train, batch_size=5000, num_workers=2)
    test_loader = DataLoader(mnist_test, batch_size=5000, num_workers=2)

    if args.matching == "wm":
        opt_perm_i = weight_matching(
            ps,
            modelA.state_dict(),
            modelB.state_dict(),
        )
    elif args.matching == "ste":
        wandb.init(project="perm_matching", config=vars(args))
        opt_perm_i = straight_through_estimator(
            ps,
            modelA,
            modelB,
            train_loader,
            test_loader,
            F.cross_entropy,
            device,
            args,
        )
    else:
        raise ValueError(f"Unknown matching {args.matching}")

    updated_params = apply_permutation(ps, opt_perm_i, modelB.state_dict())

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
        test_loss_interp_clever
    )

    os.makedirs("./plots", exist_ok=True)
    plt.savefig(
        f"./plots/mlp_mnist_interp_{args.model_a}_{args.model_b}_{args.matching}.png",
        dpi=300,
    )


if __name__ == "__main__":
    main()
