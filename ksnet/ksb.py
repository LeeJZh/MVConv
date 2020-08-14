from typing import Sequence
from torch import set_flush_denormal
import torch.nn as nn
import torch.nn.init as init
from custom_layers.crop_layer import CropLayer
import torch

class AsymConv(nn.Conv2d):
    def __init__(self, asym_type:str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if asym_type == "hor":
            mask = torch.zeros(self.kernel_size)
            mask[self.kernel_size[0]//2, :] = 1.0
        elif asym_type == "ver":
            mask = torch.zeros(self.kernel_size)
            mask[:, self.kernel_size[1]//2] = 1.0
        elif asym_type == "lx":
            mask = torch.zeros(self.kernel_size)
            for i, j in zip(range(self.kernel_size[0]), range(self.kernel_size[1])):
                mask[i, j] = 1.0
        elif asym_type == "rx":
            mask = torch.zeros(self.kernel_size)
            for i, j in zip(range(self.kernel_size[0]), range(self.kernel_size[1])):
                mask[i, j] = 1.0
            mask = mask.rot90()
        else:
            raise ValueError

        self.mask = nn.Parameter(mask, requires_grad=False)
    
    def forward(self, x):
        weight = self.weight * self.mask
        return self._conv_forward(x, weight) 



class KSBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False,
                 use_affine=True, reduce_gamma=False, use_last_bn=False, gamma_init=None ):
        super(KSBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

            
            self.ver_conv = AsymConv(asym_type="ver", in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(out_channels)

            self.hor_conv = AsymConv(asym_type="hor", in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.hor_bn = nn.BatchNorm2d(out_channels)

            self.lx_conv = AsymConv(asym_type="lx", in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.lx_bn = nn.BatchNorm2d(out_channels)
            
            self.rx_conv = AsymConv(asym_type="rx", in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.rx_bn = nn.BatchNorm2d(out_channels)


            if reduce_gamma:
                assert not use_last_bn
                self.init_gamma(1.0 / 5)

            if use_last_bn:
                assert not reduce_gamma
                self.last_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)

            if gamma_init is not None:
                assert not reduce_gamma
                self.init_gamma(gamma_init)


    def init_gamma(self, gamma_value):
        init.constant_(self.square_bn.weight, gamma_value)
        init.constant_(self.ver_bn.weight, gamma_value)
        init.constant_(self.hor_bn.weight, gamma_value)
        init.constant_(self.lx_bn.weight, gamma_value)
        init.constant_(self.rx_bn.weight, gamma_value)
        print('init gamma of square, ver, hor, lx and rx as ', gamma_value)

    def single_init(self):
        init.constant_(self.square_bn.weight, 1.0)
        init.constant_(self.ver_bn.weight, 0.0)
        init.constant_(self.hor_bn.weight, 0.0)
        init.constant_(self.lx_bn.weight, 0.0)
        init.constant_(self.rx_bn.weight, 0.0)
        print('init gamma of square as 1, ver, hor, lx and rx as 0')

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)


            vertical_outputs = self.ver_conv(input)
            vertical_outputs = self.ver_bn(vertical_outputs)

            horizontal_outputs = self.hor_conv(input)
            horizontal_outputs = self.hor_bn(horizontal_outputs)

            leftcross_outputs = self.lx_conv(input)
            leftcross_outputs = self.lx_bn(leftcross_outputs)

            rightcross_outputs = self.rx_conv(input)
            rightcross_outputs = self.rx_bn(rightcross_outputs)
            result = square_outputs + vertical_outputs + horizontal_outputs + leftcross_outputs + rightcross_outputs
            if hasattr(self, 'last_bn'):
                return self.last_bn(result)
            return result
