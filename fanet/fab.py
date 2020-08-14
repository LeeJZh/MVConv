from torch import set_flush_denormal
import torch.nn as nn
import torch.nn.init as init
from custom_layers.crop_layer import CropLayer
import torch

class FlipConv(nn.Conv2d):
    def __init__(self, filp_direction, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if torch.__version__ != "1.6.0":
            raise NotImplementedError
        if filp_direction == "lr":
            self.flip = torch.fliplr
        elif filp_direction == "ub":
            self.flip = torch.flipub
    def forward(self, input):
        weight = self.flip(self.weight)
        return self._conv_forward(input, weight)


class FABlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False,
                 use_affine=True, reduce_gamma=False, use_last_bn=False, gamma_init=None ):
        super(FABlock, self).__init__()
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

            self.ver_conv = FlipConv(filp_direction="lr", in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                      stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = FlipConv(flip_direction="ub", in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                      stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

            if reduce_gamma:
                assert not use_last_bn
                self.init_gamma(1.0 / 3)

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
        print('init gamma of square, ver and hor as ', gamma_value)

    def single_init(self):
        init.constant_(self.square_bn.weight, 1.0)
        init.constant_(self.ver_bn.weight, 0.0)
        init.constant_(self.hor_bn.weight, 0.0)
        print('init gamma of square as 1, ver and hor as 0')

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            vertical_outputs = input
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = input
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            result = square_outputs + vertical_outputs + horizontal_outputs
            if hasattr(self, 'last_bn'):
                return self.last_bn(result)
            return result
