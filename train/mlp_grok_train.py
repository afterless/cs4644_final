import os
import sys
import argparse
import random
import torch as t
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.mlp_grok import MLP
from utils.training import train, test

# Fixed params for testing
p = 256
fn = lambda x, y: (x + y) % p
d_model = 32
d_vocab = p


def cross_entropy_high_precision(logits, labels):
    """
    Shapes: batch x vocab, batch
    Cast logits to float64 because log_softmax has a float32 underflow on
    overly confident data and can only return multiples of 1.2e-7 (the smallest
    float x such that 1+x is different from 1 in float32). This leads to loss spikes
    and dodgy gradients
    """
    logprobs = F.log_softmax(logits.to(t.float64), dim=-1)
    pred_logprobs = t.gather(logprobs, dim=-1, index=labels[:, None])  # type: ignore
    loss = -t.mean(pred_logprobs)
    return loss


def gen_train_test(frac_train, num, seed=0):
    # Generate train and test split
    pairs = [((i, j), fn(i, j)) for i in range(num) for j in range(num)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train * len(pairs))
    return pairs[:div], pairs[div:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frac_train", type=float, default=0.3)
    parser.add_argument("--num_epochs", type=int, default=50000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--opt", type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=8)
    parser.add_argument("--stopping_thresh", type=float, default=3e-6)

    args = parser.parse_args()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    t.manual_seed(args.seed)
    t.cuda.manual_seed(args.seed)
    train_pairs, test_pairs = gen_train_test(args.frac_train, p, seed=args.seed)
    train_batch_size = len(train_pairs)
    test_batch_size = len(test_pairs)

    train_pairs = TensorDataset(
        t.tensor([t[0] for t in train_pairs], dtype=t.long),
        t.tensor([t[1] for t in train_pairs], dtype=t.long),
    )
    train_loader = DataLoader(
        train_pairs, batch_size=train_batch_size, shuffle=True, num_workers=2
    )

    test_pairs = TensorDataset(
        t.tensor([t[0] for t in test_pairs], dtype=t.long),
        t.tensor([t[1] for t in test_pairs], dtype=t.long),
    )
    test_loader = DataLoader(
        test_pairs, batch_size=test_batch_size, shuffle=False, num_workers=2
    )

    model = MLP(d_vocab, d_model)
    model.to(device)

    if args.opt == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98)
        )
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    else:
        raise ValueError(f"Unknown optimizer {args.opt}")
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 10, 1))

    wandb.init(
        project="git_rebasin_grok",
        entity="afterless",
        tags=["mlp", "grok", "training"],
        job_type="train",
        config=vars(args),
    )
    wandb.watch(model, log="all")
    os.makedirs("./checkpoints/mlp_grok", exist_ok=True)
    for epoch in range(1, args.num_epochs + 1):
        train(
            model,
            device,
            train_loader,
            cross_entropy_high_precision,
            optimizer,
            epoch,
            args,
        )
        test_loss, _ = test(
            model, device, epoch, test_loader, cross_entropy_high_precision
        )
        if test_loss < args.stopping_thresh:
            break
        scheduler.step()
        if epoch % args.save_every == 0:
            t.save(
                model.state_dict(),
                f"./checkpoints/mlp_grok/mlp_grok_{epoch}_{args.seed}.pt",
            )

    t.save(model.state_dict(), f"./checkpoints/mlp_grok/mlp_grok_final_{args.seed}.pt")


if __name__ == "__main__":
    main()
