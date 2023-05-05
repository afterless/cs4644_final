import os
import sys
import argparse
import torch as t
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import transforms
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.mlp_mnist import MLP
from utils.training import train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_interval", type=int, default=8)
    parser.add_argument("--stopping_thresh", type=float, default=3e-6)

    args = parser.parse_args()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    t.manual_seed(args.seed)
    t.cuda.manual_seed(args.seed)

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

    model = MLP()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs("./checkpoints/mlp_mnist", exist_ok=True)
    wandb.init(project="mlp_mnist", config=vars(args))
    wandb.watch(model, log="all")
    for epoch in range(1, args.num_epochs + 1):
        train(
            model,
            device,
            train_loader,
            F.cross_entropy,
            optimizer,
            epoch,
            args,
        )

        test_loss, _ = test(model, device, epoch, test_loader, F.cross_entropy)
        if test_loss < args.stopping_thresh:
            break
        if epoch % args.save_every == 0:
            t.save(
                model.state_dict(),
                f"./checkpoints/mlp_mnist/mlp_mnist_{epoch}_{args.seed}.pt",
            )

    t.save(
        model.state_dict(), f"./checkpoints/mlp_mnist/mlp_mnist_final_{args.seed}.pt"
    )


if __name__ == "__main__":
    main()
