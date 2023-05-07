import torch as t
import torch.nn as nn
import torch.nn.functional as F

from utils.hook_point import HookPoint


class MLP(nn.Module):
    def __init__(self, input=28 * 28):
        super().__init__()
        self.layer0 = nn.Linear(input, 512)
        self.layer1 = nn.Linear(512, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(512, 10)

        self.hook0 = HookPoint()
        self.hook1 = HookPoint()
        self.hook2 = HookPoint()
        self.hook3 = HookPoint()
        self.hook4 = HookPoint()

        self.hook0.give_name("layer0")
        self.hook1.give_name("layer1")
        self.hook2.give_name("layer2")
        self.hook3.give_name("layer3")
        self.hook4.give_name("layer4")

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.hook0(self.layer0(x)))
        x = F.relu(self.hook1(self.layer1(x)))
        x = F.relu(self.hook2(self.layer2(x)))
        x = F.relu(self.hook3(self.layer3(x)))
        x = self.hook4(self.layer4(x))
        return F.log_softmax(x, dim=-1)
