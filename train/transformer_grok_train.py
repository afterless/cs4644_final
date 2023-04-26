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
from models.transformer import Transformer
from utils.training import train, test

# Fixed params for testing
p = 113
fn = lambda x, y: (x + y) % p
d_model = 128
num_layers = 1
d_vocab = p + 1
n_ctx = 3
d_mlp = 4 * d_model
num_heads = 4
assert d_model % num_heads == 0
d_head = d_model // num_heads
act_type = "ReLU"  # Could be "GeLU"


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
    # Should I be worried about train/test overlap?
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
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--opt", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=5)

    args = parser.parse_args()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    train_pairs, test_pairs = gen_train_test(args.frac_train, p, seed=args.seed)

    train_pairs = TensorDataset(
        t.tensor([t[0] for t in train_pairs], dtype=t.long),
        t.tensor([t[1] for t in train_pairs], dtype=t.long),
    )
    train_loader = DataLoader(
        train_pairs, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    test_pairs = TensorDataset(
        t.tensor([t[0] for t in test_pairs], dtype=t.long),
        t.tensor([t[1] for t in test_pairs], dtype=t.long),
    )
    test_loader = DataLoader(
        test_pairs, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    model = Transformer(
        num_layers, d_vocab, d_model, d_mlp, d_head, num_heads, n_ctx, act_type
    )
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

    wandb.init(project="transformer_mod_grok", config=vars(args))
    wandb.watch(model, log="all")
    os.makedirs("./checkpoints", exist_ok=True)
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
        test(model, device, epoch, test_loader, cross_entropy_high_precision)
        scheduler.step()
        if epoch % args.save_every == 0:
            t.save(model.state_dict(), f"./checkpoints/transformer_grok_{epoch}.pt")


if __name__ == "__main__":
    main()
