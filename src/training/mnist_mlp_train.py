import numpy as np
import tensorflow as tf
import wandb
import pickle
from tqdm import tqdm
import weight_matching
from models.mlp import MLP
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

def mnist_mlp_train(num_models=2):
    training_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
    )

    test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(device)
    # print(training_data[0][0].shape)
    # print(training_data[0][0].squeeze().shape)
    # print(plt.imshow(training_data[0][0].squeeze(), cmap="gray"))
    # print(training_data[0][1])
    # print(len(training_data))
    # figure = plt.figure(figsize=(8,8))
    # cols, rows = 4, 2
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(training_data), size=(1,)).item()
    #     img, label = training_data[sample_idx]
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(label)
    #     plt.axis("off")
    #     plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()
    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)
    # print(len(test_dataloader))

    models_to_save = {}
    for i in range(num_models):
        model = MLP().to(device)
        # print(model)
        lr = 1e-3
        bs = 64
        epochs = 8
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for e in range(epochs):
            print("Epoch:", e + 1)
            for xb, yb in train_dataloader:
                preds = model(xb)
                loss = loss_fn(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss = loss.item()
            print(f"Train loss: {loss:>7f}")

        num_batches = len(test_dataloader)
        size = len(test_dataloader.dataset)
        test_loss, corrects = 0, 0

        with torch.no_grad():
            for xb, yb in test_dataloader:
                preds = model(xb)
                test_loss += loss_fn(preds, yb).item()
                corrects += (preds.argmax(1) == yb).type(torch.float).sum().item()
        
        test_loss /= num_batches
        corrects /= size
        print(f"Test loss: \n Accuracy: {(100*corrects):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        # path = "training/saved_models/mnist_mlp/model" + str(i + 1) + ".pt"
        # torch.save(model, path)
        models_to_save[str('model_' + str(i+1) + "_state_dict")] = model.state_dict()
        models_to_save[str('optimizer_' + str(i+1) + "_state_dict")] = optimizer.state_dict()
        models_to_save['model_' + str(i+1) + '_lr'] = lr
        models_to_save['model_' + str(i+1) + '_bs'] = bs
        models_to_save['model_' + str(i+1) + '_loss_type'] = "nn.CrossEntropyLoss"
    models_to_save["training_data"] = training_data
    models_to_save["test_data"] = test_data
    # Make an enum for model type?
    models_to_save["model_type"] = "MLP"
    path = "training/saved_models/mnist_mlp/" + str(num_models) + "models.pt"
    torch.save(models_to_save, path)