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


# Asier Moment
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


def resnet50_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None)}
    norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}
    dense = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,),
    }

    # These are blocks that have a residual connection, but don't change the number of channels
    easyblock = lambda name, p: {
        **norm(f"{name}.bn1.norm", p),
        **conv(f"{name}.conv1", p, f"P_{name}_inner"),
        **norm(f"{name}.bn2", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p),
    }

    # For blocks that change channels, we need to add a shortcut
    shortcutblock = lambda name, p_in, p_out: {
        **norm(f"{name}.bn1", p_in),
        **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **norm(f"{name}.bn2", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
        **conv(f"{name}.shortcut.0", p_in, p_out),
        **norm(f"{name}.shortcut.1", p_out),
    }

    return permutation_spec_from_axes_to_perm(
        {
            **conv(
                "conv1", None, "P_bg0"
            ),  # bottleneck layer (or aggresive downsampling)
            **shortcutblock("layer1.0", "P_bg0", "P_bg1"),
            **easyblock("layer1.1", "P_bg1"),
            **easyblock("layer1.2", "P_bg1"),
            **easyblock("layer1.3", "P_bg1"),
            **easyblock("layer1.4", "P_bg1"),
            **easyblock("layer1.5", "P_bg1"),
            **easyblock("layer1.6", "P_bg1"),
            **easyblock("layer1.7", "P_bg1"),
            **shortcutblock("layer2.0", "P_bg1", "P_bg2"),
            **easyblock("layer2.1", "P_bg2"),
            **easyblock("layer2.2", "P_bg2"),
            **easyblock("layer2.3", "P_bg2"),
            **easyblock("layer2.4", "P_bg2"),
            **easyblock("layer2.5", "P_bg2"),
            **easyblock("layer2.6", "P_bg2"),
            **easyblock("layer2.7", "P_bg2"),
            **shortcutblock("layer3.0", "P_bg2", "P_bg3"),
            **easyblock("layer3.1", "P_bg3"),
            **easyblock("layer3.2", "P_bg3"),
            **easyblock("layer3.3", "P_bg3"),
            **easyblock("layer3.4", "P_bg3"),
            **easyblock("layer3.5", "P_bg3"),
            **easyblock("layer3.6", "P_bg3"),
            **easyblock("layer3.7", "P_bg3"),
            **norm("bn1", "P_bg3"),
            **dense("fc", "P_bg3", None),
        }
    )


def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """
    Given a permutation, a parameter name, and a set of parameters, returns the permuted parameter
    """
    w = params[k]
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
    Given two sets of parameters, returns a permutation that matches param_a with param_b. Algo #2/#3
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
