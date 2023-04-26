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

def straight_through_estimator_training(ps: weight_matching.PermutationSpec, modelA_parameters, modelB_parameters,
                                        training_data, test_data, lr = 1e-3, bs = 64, loss_type = "nn.CrossEntropyLoss", model_type = "MLP"):
    train_dataloader = DataLoader(training_data, batch_size=bs)
    test_dataloader = DataLoader(test_data, batch_size=bs)
    # for a in modelA_parameters:
    #     print(a)
    #     print(modelA_parameters[a])
    # This method currently only supports MLP models
    if(model_type == "MLP"):
        model = MLP()
        phi_model = MLP()
    else:
        raise ValueError
    model_state_dict = model.state_dict()
    phi_state_dict = phi_model.state_dict()
    # This loop for initializing model's parameters to modelA's parameters is wrong, and needs updating
    model_state_dict.update(modelA_parameters)
    model.load_state_dict(model_state_dict)
    # for m, a in zip(model_state_dict.items(), modelA_parameters):
        # new_param = a[1]
        # m[1].copy_(new_param)
        # model_state_dict[m] = modelA_parameters[a]
    # Yet to implement epochs
    epochs = 8
    # This method currently only supports Cross Entropy Loss
    if(loss_type == "nn.CrossEntropyLoss"):
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for xb, yb in train_dataloader:
        # Need to get model's parameters correctly:
        model_state_dict = model.state_dict()
        params_model = {}
        for name, para in model.named_parameters():
            params_model[name] = para
        perm = weight_matching.weight_matching(ps, params_model, modelB_parameters)
        projected_parameters = weight_matching.apply_permutation(ps, perm, modelB_parameters)
        # for phi, proj, a in zip(phi_state_dict.items(), projected_parameters, modelA_parameters):
        #     new_param = 0.5 * (proj[1] + a[1])
        #     phi[1].copy_(new_param)
        for name, para in projected_parameters:
            print(name)
        for name, para in modelA_parameters:
            print(name)
        for (proj_name, proj_para), (a_name, a_para) in zip(projected_parameters, modelA_parameters):
            phi_state_dict[proj_name] = 0.5 (proj_para + a_para)
        # new_phi_state_dict = 0.5 * (projected_parameters + modelA_parameters)
        phi_model.load_state_dict(phi_state_dict)
        preds = phi_model(xb)
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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

    ps = weight_matching.mlp_permutation_spec(num_hidden_layers=3)
    # print(modelA)
    # print(modelA.parameters())
    # print(modelA.layers)

    # Trying to have parameters be the same as in weight_matching.py
    params_a = {}
    for name, para in modelA.named_parameters():
        params_a[name] = para
    params_b = {}
    for name, para in modelB.named_parameters():
        params_b[name] = para

    # Testcall for straight_through_estimator_training
    optimal_permutation_phi = straight_through_estimator_training(
        ps, params_a, params_b, training_data, test_data, lr=lr, bs=bs, model_type=model_type)