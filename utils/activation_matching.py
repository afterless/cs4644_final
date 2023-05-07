import torch as t
from scipy.optimize import linear_sum_assignment


def activation_matching(modelA, modelB, train_loader, args):
    """
    Given two models, return permutation of modelB to match modelA based on
    activations.
    """
    modelA.eval()
    modelB.eval()

    def hook(act, name):
        pass

    for pA, pB in zip(modelA, modelB):
        if "hook" in pA:
            modelA[pA].add(hook)
            modelB[pB].add(hook)

    # Get activations for modelA
    activationsA = []
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        activationsA.append(modelA(data))
        if i == 100:
            break

    # Get activations for modelB
    activationsB = []
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        activationsB.append(modelB(data))
        if i == 100:
            break

    # Compute cost matrix
    cost_matrix = []
    for i, activationA in enumerate(activationsA):
        cost_matrix.append([])
        for j, activationB in enumerate(activationsB):
            cost_matrix[i].append(t.norm(activationA - activationB).item())

    # Compute optimal permutation
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    perm = {}
    for i, j in zip(row_ind, col_ind):
        perm[f"layer{i}.weight"] = f"layer{j}.weight"
        perm[f"layer{i}.bias"] = f"layer{j}.bias"

    return perm
