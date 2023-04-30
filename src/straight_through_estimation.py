import numpy as np
import tensorflow as tf
from tqdm import tqdm
import weight_matching
import torch
import pickle
from models.mlp import MLP
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
from training import mnist_mlp_train
import re

# def param_dict_with_correct_keys(model):
#     params_model = {}
#     for name, para in model.named_parameters():
#         if(name.__contains__("r0.") or name.__contains__("N")):
#             pass
#         else:
#             name = name.replace(".","",1)
#             digits = filter(str.isdigit, name)
#             digits = list(digits)
#             replacements = []
#             for digit in digits:
#                 replacements.append(str(int(digit) + 1))
#             for i in range(len(digits)):
#                 name = name.replace(digits[i], replacements[i],1)
#             params_model[name] = para
#     return params_model

def straight_through_estimator_training(ps: weight_matching.PermutationSpec, modelA, modelB,
                                        training_data, test_data, lr = 1e-3, bs = 64, loss_type = "nn.CrossEntropyLoss", model_type = "MLP"):

    train_dataloader = DataLoader(training_data, batch_size=bs)
    test_dataloader = DataLoader(test_data, batch_size=bs)

    # This method currently only supports MLP models
    if(model_type == "MLP"):
        model = MLP()
        phi_model = MLP()
    else:
        raise ValueError

    model_state_dict = model.state_dict()
    phi_state_dict = phi_model.state_dict()

    params_a = {}
    for name, para in modelA.named_parameters():
        params_a[name] = para

    params_b = {}
    for name, para in modelB.named_parameters():
        params_b[name] = para

    params_model = {}
    for name, para in model.named_parameters():
        params_model[name] = para

    model_state_dict.update(params_a)
    model.load_state_dict(model_state_dict)

    epochs = 8
    # This method currently only supports Cross Entropy Loss
    if(loss_type == "nn.CrossEntropyLoss"):
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        for xb, yb in train_dataloader:
            perm = weight_matching.weight_matching(ps, params_model, params_b)
            projected_parameters = weight_matching.apply_permutation(ps, perm, params_b)

            for proj_name, (a_name, a_para) in zip(projected_parameters, modelA.named_parameters()):
                phi_state_dict[a_name] = 0.5 * (projected_parameters[proj_name] + a_para)
            phi_model.load_state_dict(phi_state_dict)
            preds = phi_model(xb)
            loss = loss_fn(preds, yb)
            # TODO: Idea: Compute a loss for model, add layer that changes model's loss to phi_model's
            # loss which has derivative 1, and then backpropagates.
            optimizer.zero_grad()
            loss.backward()
            for model_param, phi_model_param in zip(model.parameters(), phi_model.parameters()):
                model_param.grad = phi_model_param.grad

            # Pytorch won't let me change model parameters mid-backwards pass
            # for name, param in model.named_parameters():
            #    grad_storage[name] = param
            # model.load_state_dict(model_state_dict)
            # for name, param in model.named_parameters():
            #    param.grad = grad_storage[name]

            optimizer.step()
            for name, para in model.named_parameters():
                params_model[name] = para
            model_state_dict = model.state_dict()
        loss = loss.item()
        print(f"Train loss: {loss:>7f}")
    return perm
if __name__ == "__main__":
    lr = 1e-3
    # mnist_mlp_train.mnist_mlp_train()

    modelA = MLP()
    modelB = MLP()
    optimizerA = torch.optim.Adam(modelA.parameters(), lr=lr)
    optimizerB = torch.optim.Adam(modelB.parameters(), lr=lr)

    # Note that modelA is model_1 and modelB is model_2
    checkpoint = torch.load("training/saved_models/mnist_mlp/2models.pt")

    modelA.load_state_dict(checkpoint['model_1_state_dict'])
    modelB.load_state_dict(checkpoint['model_2_state_dict'])

    optimizerA.load_state_dict(checkpoint['optimizer_1_state_dict'])
    optimizerB.load_state_dict(checkpoint['optimizer_2_state_dict'])

    # For simplicity, use model_1's  lr, bs, and loss.
    lr = checkpoint["model_1_lr"]
    bs = checkpoint["model_1_bs"]
    loss_type = checkpoint["model_1_loss_type"]

    model_type = checkpoint["model_type"]

    training_data = checkpoint["training_data"]
    test_data = checkpoint["test_data"]

    modelA.train()
    modelB.train()

    ps = weight_matching.mlp_permutation_spec(num_hidden_layers=4)

    # Testcall for straight_through_estimator_training
    optimal_permutation_phi = straight_through_estimator_training(
        ps, modelA, modelB, training_data, test_data, lr=lr, bs=bs, model_type=model_type)