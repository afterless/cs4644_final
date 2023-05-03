import argparse
import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mlp_grok import MLP
from train.mlp_grok_train import cross_entropy_high_precision, gen_train_test
from utils import weight_matching
from utils.straight_through_estimator import straight_through_estimator

# Fixed params for testing
p = 113
d_model = 64
d_vocab = p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--frac_train", type=float, default=0.3)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    chkptA = t.load("../train/checkpoints/mlp_grok_final_1.pt", map_location=device)
    chkptB = t.load("../train/checkpoints/mlp_grok_final_2.pt", map_location=device)

    modelA = MLP(d_vocab, d_model)
    modelA.to(device)
    modelA.load_state_dict(chkptA)

    modelB = MLP(d_vocab, d_model)
    modelB.to(device)
    modelB.load_state_dict(chkptB)

    ps = weight_matching.mlp_grok_permutation_spec(num_hidden_layers=2)
    train_pairs, test_pairs = gen_train_test(args.frac_train, p, seed=args.seed)

    train_pairs = TensorDataset(
        t.tensor([t[0] for t in train_pairs], dtype=t.long),
        t.tensor([t[1] for t in train_pairs], dtype=t.long),
    )

    train_loader = DataLoader(
        train_pairs, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    opt_perm_pi = straight_through_estimator(
        ps, modelA, modelB, train_loader, cross_entropy_high_precision, device, args
    )
