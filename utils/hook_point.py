import torch.nn as nn


"""
A helper class thata gets access to immediate activations of a layer
The idea is that you can wrap any intermediate activation in a HookPoint and get a convenient
way to add PyTorch hooks
"""


class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []

    def give_name(self, name):
        # Called by model at init
        self.name = name

    def add_hook(self, hook, dir="fwd"):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format fn(module, input, output)

        def full_hook(module, module_input, module_output):
            return hook(module_output, self.name)

        if dir == "fwd":
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir == "bwd":
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Unknown direction {dir}")

    def remove_hooks(self, dir="fwd"):
        if dir == "fwd" or dir == "both":
            for handle in self.fwd_hooks:
                handle.remove()
            self.fwd_hooks = []
        if dir == "bwd" or dir == "both":
            for handle in self.bwd_hooks:
                handle.remove()
            self.bwd_hooks = []
        if dir not in ["fwd", "bwd", "both"]:
            raise ValueError(f"Unknown direction {dir}")

    def forward(self, x):
        return x
