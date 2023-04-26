import torch.nn as nn
import torch.nn.functional as F


# TODO: Needs to be fixed to match spec
class MLP(nn.Module):
    def __init__(self, input=28 * 28):
        super().__init__()
        self.layer0 = nn.Linear(input, 512)
        self.layers = nn.ModuleList([nn.Linear(512, 512) for _ in range(3)])
        self.layerN = nn.Linear(512, 10)
        # Quentin: I added a log softmax layer to match Ainsworth et al's implementation
        self.LogSoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.layer0(x))
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        x = F.relu(self.layerN(x))
        # Quentin: I changed the return to a log softmax to match Ainsworth et al's implementation
        return self.LogSoftmax(x)
