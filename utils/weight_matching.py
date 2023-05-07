from collections import defaultdict
from typing import NamedTuple

import torch as t
from scipy.optimize import linear_sum_assignment


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def mlp_permutation_spec(num_hidden_layers: int) -> PermutationSpec:
    assert num_hidden_layers >= 1
    return permutation_spec_from_axes_to_perm(
        {
            "layer0.weight": ("P_0", None),
            **{
                f"layer{i}.weight": (f"P_{i}", f"P_{i-1}")
                for i in range(1, num_hidden_layers)
            },
            **{f"layer{i}.bias": (f"P_{i}",) for i in range(num_hidden_layers)},
            f"layer{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers-1}"),
            f"layer{num_hidden_layers}.bias": (None,),
        }
    )


def mlp_grok_permutation_spec(num_hidden_layers: int) -> PermutationSpec:
    assert num_hidden_layers >= 1
    return permutation_spec_from_axes_to_perm(
        {
            "embed.W_E": (None, "P_0"),
            "layer0.weight": ("P_1", None),
            **{
                f"layer{i}.weight": (f"P_{i+1}", f"P_{i}")
                for i in range(1, num_hidden_layers)
            },
            "unembed.W_U": (f"P_{num_hidden_layers}", None),
        }
    )


def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """
    Given a permutation, a parameter name, and a set of parameters, returns the permuted parameter
    """
    w = params[k]
    if k in ps.axes_to_perm:
        for axis, p in enumerate(ps.axes_to_perm[k]):
            # Skip axis we're trying to permute
            if axis == except_axis:
                continue

            if p is not None:
                w = t.index_select(w, axis, perm[p].int())

    return w


def apply_permutation(ps: PermutationSpec, perm, params):
    """
    Given a permutation and a set of parameters, returns a new set of parameters with the same structure, but permuted
    """
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def weight_matching(
    ps: PermutationSpec, params_a, params_b, max_iter=100, init_perm=None
):
    """
    Given two sets of parameters, returns a permutation of param_b to match with param_a Algo #2/#3
    """
    perm_sizes = {
        p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()
    }

    perm = (
        {p: t.arange(n) for p, n in perm_sizes.items()}
        if init_perm is None
        else init_perm
    )
    perm_names = list(perm.keys())

    for i in range(max_iter):
        progress = False
        for p_ix in t.randperm(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = t.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                w_a = t.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = t.moveaxis(w_b, axis, 0).reshape((n, -1))

                A += t.matmul(w_a, w_b.T)

            ri, ci = linear_sum_assignment(A.detach().numpy(), maximize=True)
            assert (t.tensor(ri) == t.arange(len(ri))).all()
            oldL = t.einsum("ij,ij->i", A, t.eye(n)[perm[p].long()]).sum()
            newL = t.einsum("ij,ij->i", A, t.eye(n)[ci, :]).sum()
            print(f"{i}/{p}: {newL - oldL}")
            progress = progress or newL - oldL > 1e-12

            perm[p] = t.tensor(ci)

        if not progress:
            break

    return perm


def test_weight_matching():
    ps = mlp_permutation_spec(num_hidden_layers=1)
    print(ps.axes_to_perm)
    rng = t.Generator()
    rng.manual_seed(1746)
    num_hidden = 10
    shapes = {
        "layer0.weight": (2, num_hidden),
        "layer0.bias": (num_hidden,),
        "layer1.weight": (num_hidden, 3),
        "layer1.bias": (3,),
    }

    params_a = {k: t.randn(*v, generator=rng) for k, v in shapes.items()}
    params_b = {k: t.randn(*v, generator=rng) for k, v in shapes.items()}
    perm = weight_matching(ps, params_a, params_b)
    print(perm)


if __name__ == "__main__":
    test_weight_matching()
