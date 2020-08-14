import torch.nn as nn
import torch.nn.init as init
from custom_layers.channel_dropout_layer import ChannelDropLayer
from custom_layers.clamp_conv1x1 import ClampConv
from custom_layers.diff_reg_layer import DiffRegLayer

PROJECTION_SCALE = 2

class RFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False,
                 use_affine=True, reduce_gamma=False, use_last_bn=False, gamma_init=None, alpha=1e-4, scale=2):
        super(RFBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            # self.main_drop = ChannelDropLayer()
            # self.main_alter = ClampConv(
            #     in_channels=in_channels, out_channels=in_channels//3, kernel_size=1, bias=False)

            if in_channels // scale < 8:
                print("in channnels {} with scale {} too small".format(in_channels, scale))
                scale = 1
                self.alpha = 0
            else:
                self.alpha = 1e-4

            self.main_alter = nn.Identity()
            self.main_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=(
                                           kernel_size, kernel_size), stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=False,
                                       padding_mode=padding_mode)
            self.main_bn = nn.BatchNorm2d(
                num_features=out_channels, affine=use_affine)

            # self.left_drop = ChannelDropLayer()
            self.left_alter = nn.Conv2d(
                in_channels, in_channels//scale, kernel_size=1, bias=False)
            # self.right_drop = ChannelDropLayer()
            self.right_alter = nn.Conv2d(
                in_channels, in_channels//scale, kernel_size=1, bias=False)

            self.diff_reg = DiffRegLayer()
            self.left_conv = nn.Conv2d(in_channels=in_channels//scale, out_channels=out_channels,
                                       kernel_size=(kernel_size, kernel_size),
                                       stride=stride, padding=padding,
                                       dilation=dilation, groups=groups, bias=False,
                                       padding_mode=padding_mode)
            self.left_bn = nn.BatchNorm2d(num_features=out_channels)

            self.right_conv = nn.Conv2d(in_channels=in_channels//scale, out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size),
                                        stride=stride, padding=padding,
                                        dilation=dilation, groups=groups, bias=False,
                                        padding_mode=padding_mode)
            self.right_bn = nn.BatchNorm2d(num_features=out_channels)

            self.save_left = None
            self.save_right = None

            if reduce_gamma:
                assert not use_last_bn
                self.init_gamma(1.0 / 3)

            if use_last_bn:
                assert not reduce_gamma
                self.last_bn = nn.BatchNorm2d(
                    num_features=out_channels, affine=True)

            if gamma_init is not None:
                assert not reduce_gamma
                self.init_gamma(gamma_init)

    def init_gamma(self, gamma_value):
        init.constant_(self.main_bn.weight, gamma_value)
        init.constant_(self.left_bn.weight, gamma_value)
        init.constant_(self.right_bn.weight, gamma_value)
        print('init gamma of main, left and right as ', gamma_value)

    def single_init(self):
        init.constant_(self.main_bn.weight, 1.0)
        init.constant_(self.left_bn.weight, 0.0)
        init.constant_(self.right_bn.weight, 0.0)
        print('init gamma of main as 1, left and right as 0')

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:

            main_outputs = input
            main_outputs = self.main_alter(main_outputs)
            main_outputs = self.main_conv(main_outputs)
            main_outputs = self.main_bn(main_outputs)

            left_outputs = input
            left_outputs = self.left_alter(left_outputs)
            right_outputs = input
            right_outputs = self.right_alter(right_outputs)

            # if self.training:
            #     self.save_left = left_outputs
            #     self.save_right = right_outputs
            # left_outputs, right_outputs = self.diff_reg(left_outputs, right_outputs)

            right_outputs = self.right_conv(right_outputs)
            right_outputs = self.right_bn(right_outputs)
            left_outputs = self.left_conv(left_outputs)
            left_outputs = self.left_bn(left_outputs)
            result = main_outputs + left_outputs + right_outputs
            if hasattr(self, 'last_bn'):
                return self.last_bn(result)
            return result
