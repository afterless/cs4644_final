import torch.nn as nn
import torch.nn.functional as F


# TODO: Needs to be fixed to match spec
class MLP(nn.Module):
    def __init__(self, input=28 * 28):
        super().__init__()
        self.layer0 = nn.Linear(input, 512)
        # self.layer = nn.ModuleList([nn.Linear(512, 512) for i in range(3)])
        self.layer1 = nn.Linear(512, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 512)
        # Change name of layerN to layer4
        # self.layerN = nn.Linear(512, 10)
        self.layer4 = nn.Linear(512, 10)
        # Quentin: I added a log softmax layer to match Ainsworth et al's implementation
        self.LogSoftmax = nn.LogSoftmax()

    # def hidden_layer_naming(layer):

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.layer0(x))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        # Quentin: I changed the return to a log softmax to match Ainsworth et al's implementation
        return self.LogSoftmax(x)
