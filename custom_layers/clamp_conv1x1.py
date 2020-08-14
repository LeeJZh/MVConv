import torch
from torch import nn

BOUND = 1.0

class ClampConv(nn.Conv2d):
    def __init__(self, *arg, **kwarg) -> None:
        super(ClampConv, self).__init__(*arg, **kwarg)
        self.bound = BOUND
    
    def forward(self, x):
        # weight = torch.clamp(self.weight, min=-self.bound, max=self.bound)
        # return self._conv_forward(x, weight)
        return self._conv_forward(x, self.weight)