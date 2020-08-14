import torch
from torch import nn


class ChannelDropLayer(nn.Module):
    def __init__(self) -> None:
        super(ChannelDropLayer, self).__init__()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        else:
            mask = torch.randint(low=0, high=2, size=(x.size(1), )).view(1, -1, 1, 1).cuda()
            return x * mask / (1 - 0.5)
