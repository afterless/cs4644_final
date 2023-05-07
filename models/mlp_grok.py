import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, rearrange
from utils.hook_point import HookPoint


class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.parameter.Parameter(t.randn(d_model, d_vocab) / d_model**0.5)

    def forward(self, x: t.LongTensor):
        return t.einsum("dbp->bpd", self.W_E[:, x])


class UnEmbed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.parameter.Parameter(t.randn(d_model, d_vocab) / d_vocab**0.5)

    def forward(self, x):
        return t.matmul(x, self.W_U)


# TODO: Needs to be fixed to match spec
class MLP(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.embed = Embed(d_vocab, d_model)
        self.layer0 = nn.Linear(d_model + d_model, d_model * 2, bias=False)
        self.layer1 = nn.Linear(d_model * 2, d_model, bias=False)
        self.unembed = UnEmbed(d_vocab, d_model)
        self.hook0 = HookPoint()
        self.hook1 = HookPoint()
        self.hook2 = HookPoint()
        self.hook3 = HookPoint()

    def forward(self, x):
        x = self.hook0(self.embed(x))
        x = rearrange(x, "b s d -> b (s d)")
        # x = x.sum(dim=1)
        x = F.relu(self.hook1(self.layer0(x)))
        x = F.relu(self.hook2(self.layer1(x)))
        x = self.hook3(self.unembed(x))
        return F.log_softmax(x, dim=-1)
