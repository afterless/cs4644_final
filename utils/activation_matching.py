from collections import defaultdict
from typing import NamedTuple
from tqdm.auto import tqdm

import torch as t
from scipy.optimize import linear_sum_assignment


def activation_matching(ps, modelA, modelB, train_loader, device):
    """
    Given two models, return permutation of modelB to match modelA based on
    activations.
    """
    modelA.eval()
    modelB.eval()
    activationsA = defaultdict(float)
    activationsB = defaultdict(float)
    paramsA = modelA.state_dict()

    def hookA(act, name):
        activationsA[name] += act
        pass

    def hookB(act, name):
        activationsB[name] += act
        pass

    for nA, nB in zip(modelA.named_children(), modelB.named_children()):
        if "hook" in nA[0]:
            nA[1].add_hook(hookA)
            nB[1].add_hook(hookB)

    for i, (data, _) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        modelA(data)
        modelB(data)
        if i == 1:
            break

    # Might be unecessary
    for (nA, actA), (nB, actB) in zip(activationsA.items(), activationsB.items()):
        activationsA[nA] = actA / len(train_loader)
        activationsB[nB] = actB / len(train_loader)

    perm_sizes = {
        p: paramsA[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()
    }

    perm = {p: t.arange(n) for p, n in perm_sizes.items()}
    perm_names = list(perm.keys())

    for p_ix in tqdm(range(len(perm_names))):
        p = perm_names[p_ix]
        n = perm_sizes[p]

        A = t.zeros((n, n))
        for wk, axis in ps.perm_to_axes[p]:
            if axis != 0 or wk.split(".")[0] not in activationsA:
                continue
            a_a = activationsA[wk.split(".")[0]]
            a_b = activationsB[wk.split(".")[0]]
            # a_a = a_a.reshape((a_a.shape[0], -1))
            # a_b = a_b.reshape((a_b.shape[0], -1))
            A += t.matmul(a_a.T, a_b)  # type: ignore

        ri, ci = linear_sum_assignment(A.detach().numpy(), maximize=True)
        assert (t.tensor(ri) == t.arange(len(ri))).all()
        perm[p] = t.tensor(ci)

    return perm
