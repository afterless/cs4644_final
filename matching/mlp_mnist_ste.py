import argparse
import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import transforms
import wandb

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mlp_mnist import MLP
from utils.weight_matching import (
    apply_permutation,
    mlp_permutation_spec,
)
from utils.straight_through_estimator import straight_through_estimator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1.0)

    args = parser.parse_args()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    chkptA = t.load(
        "../train/checkpoints/mlp_mnist/mlp_mnist_final_1.pt", map_location=device
    )
    chkptB = t.load(
        "../train/checkpoints/mlp_mnist/mlp_mnist_final_3.pt", map_location=device
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
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    mnist_train = torchvision.datasets.MNIST(
        "./mnist_data_train", train=True, download=True, transform=transform
    )
    mnist_test = torchvision.datasets.MNIST(
        "./mnist_data_test", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False, num_workers=2)
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
