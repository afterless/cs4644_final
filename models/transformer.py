import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from utils.hook_point import HookPoint

"""
If this looks verbose, it's because it is. It helps with understanding what's going on
and some interpretability stuff, thx Neel :). Might rewrite this if unecessary, after deadline.
"""


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


class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_P = nn.parameter.Parameter(t.randn(max_ctx, d_model) / d_model**0.5)

    def forward(self, x: t.LongTensor):
        return x + self.W_P[: x.shape[-2]]


# LayerNorm (may remove, infamously makes interp hard)
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6, model=[None]):
        super().__init__()
        self.model = model
        self.gamma = nn.parameter.Parameter(t.ones(d_model))
        self.beta = nn.parameter.Parameter(t.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.eps)
            x = x * self.gamma + self.beta
            return x
        else:
            return x


# Attention
class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_Q = nn.parameter.Parameter(
            t.randn(num_heads, d_head, d_model) / d_model**0.5
        )
        self.W_K = nn.parameter.Parameter(
            t.randn(num_heads, d_head, d_model) / d_model**0.5
        )
        self.W_V = nn.parameter.Parameter(
            t.randn(num_heads, d_head, d_model) / d_model**0.5
        )
        self.W_O = nn.parameter.Parameter(
            t.randn(d_model, d_head * num_heads) / d_model**0.5
        )
        self.register_buffer("mask", t.triu(t.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x):
        q = t.einsum("ihd,bpd->biph", self.W_Q, x)
        k = t.einsum("ihd,bpd->biph", self.W_K, x)
        v = t.einsum("ihd,bpd->biph", self.W_V, x)
        attn_scores_pre = t.einsum("biph,biqh->biqp", k, q)
        attn_scores_masked = t.tril(attn_scores_pre) - 1e10 * (1 - self.mask[: x.shape[-2], : x.shape[-2]])  # type: ignore
        attn_scores = self.hook_attn(
            F.softmax(
                self.hook_attn_pre(attn_scores_masked / self.d_head**0.5), dim=-1
            )
        )
        z = self.hook_z(t.einsum("biph,biqp->biqh", v, attn_scores))
        z_flat = rearrange(z, "b i q h -> b q (i h)")
        out = t.einsum("df,bqf->bqd", self.W_O, z_flat)
        return out


# MLP
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.parameter.Parameter(t.randn(d_mlp, d_model) / d_model**0.5)
        self.b_in = nn.parameter.Parameter(t.zeros(d_mlp))
        self.W_out = nn.parameter.Parameter(t.randn(d_model, d_mlp) / d_model**0.5)
        self.b_out = nn.parameter.Parameter(t.zeros(d_model))
        self.act_type = act_type

        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        assert act_type in ["ReLU", "GeLU"]

    def forward(self, x):
        x = self.hook_pre(t.einsum("md,bpd->bpm", self.W_in, x) + self.b_in)
        if self.act_type == "ReLU":
            x = F.relu(x)
        elif self.act_type == "GeLU":
            x = F.gelu(x)
        x = self.hook_post(x)
        x = t.einsum("dm,bpm->bpd", self.W_out, x) + self.b_out
        return x


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        self.model = model
        # self.ln1 = LayerNorm(d_model, model=self.model), add or remove if necessary
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        # self.ln2 = LayerNorm(d_model, model=self.model), add or remove if necessary
        self.mlp = MLP(d_model, d_mlp, act_type, model=self.model)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x):
        x = self.attn(self.hook_resid_pre(x))
        x = self.hook_resid_mid(x + self.hook_attn_out(x))
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp(x)))
        return x


# Full Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        num_layers,
        d_vocab,
        d_model,
        d_mlp,
        d_head,
        num_heads,
        n_ctx,
        act_type,
        use_cache=False,
    ):
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache

        self.embed = Embed(d_vocab, d_model)
        self.pos_emb = PosEmbed(n_ctx, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model=[self]
                )
                for _ in range(num_layers)
            ]
        )
        self.unembed = UnEmbed(d_vocab, d_model)

        for n, m in self.named_modules():
            if type(m) == HookPoint:
                m.give_name(n)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_emb(x)
        for b in self.blocks:
            x = b(x)
        x = self.unembed(x)
        return x

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def hook_points(self):
        return [m for n, m in self.named_modules() if "hook" in n]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks("fwd")  # type: ignore
            hp.remove_hooks("bwd")  # type: ignore

    def cache_all(self, cache, incl_bwd=False):
        # Cache all activations in a HookPoint
        def save_hook(t, n):
            cache[n] = t.detach()

        def save_hook_back(t, n):
            cache[n + "_grad"] = t[0].detach()

        for hp in self.hook_points():
            hp.add_hook(save_hook, "fwd")  # type: ignore
            if incl_bwd:
                hp.add_hook(save_hook_back, "bwd")  # type: ignore
