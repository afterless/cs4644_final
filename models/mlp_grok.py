import torch as t
import torch.nn as nn
import torch.nn.functional as F


# Not sure why positional embedding doesn't exist in pytorch?
class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_P = nn.parameter.Parameter(t.randn(max_ctx, d_model) / d_model**0.5)

    def forward(self, x: t.LongTensor):
        return x + self.W_P[: x.shape[-2]]


# TODO: Needs to be fixed to match spec
class MLP(nn.Module):
    def __init__(self, d_vocab, max_ctx, d_model):
        super().__init__()
        self.embed = nn.Embedding(d_vocab, d_model)
        self.pos_embed = PosEmbed(max_ctx, d_model)
        self.layer0 = nn.Linear(d_model, d_model, bias=False)
        self.unembed = nn.Linear(d_model, d_vocab, bias=False)

    def forward(self, x):
        x = self.embed(x)
        x += self.pos_embed(x)
        x = F.relu(self.layer0(x))
        x = self.unembed(x)
        return x
