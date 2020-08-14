from utils.misc import read_hdf5, save_hdf5
import numpy as np

SQUARE_KERNEL_KEYWORD = 'square_conv.weight'

def _fuse_kernel(kernel, gamma, std):
    b_gamma = np.reshape(gamma, (kernel.shape[0], 1, 1, 1))
    b_gamma = np.tile(b_gamma, (1, kernel.shape[1], kernel.shape[2], kernel.shape[3]))
    b_std = np.reshape(std, (kernel.shape[0], 1, 1, 1))
    b_std = np.tile(b_std, (1, kernel.shape[1], kernel.shape[2], kernel.shape[3]))
    return kernel * b_gamma / b_std

def _add_to_square_kernel(square_kernel, asym_kernel):
    square_kernel[:] += asym_kernel

def convert_ksnet_weights(train_weights, deploy_weights, eps):
    train_dict = read_hdf5(train_weights)
    print(train_dict.keys())
    deploy_dict = {}
    square_conv_var_names = [name for name in train_dict.keys() if SQUARE_KERNEL_KEYWORD in name]
    for square_name in square_conv_var_names:
        square_kernel = train_dict[square_name]
        square_mean = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.running_mean')]
        square_std = np.sqrt(train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.running_var')] + eps)
        square_gamma = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.weight')]
        square_beta = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.bias')]

        ver_kernel = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_conv.weight')]
        ver_mask = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_conv.mask')]
        ver_kernel = ver_kernel * ver_mask
        ver_mean = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_bn.running_mean')]
        ver_std = np.sqrt(train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_bn.running_var')] + eps)
        ver_gamma = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_bn.weight')]
        ver_beta = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_bn.bias')]

        hor_kernel = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_conv.weight')]
        hor_mask = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_conv.mask')]
        hor_kernel = hor_kernel * hor_mask
        hor_mean = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_bn.running_mean')]
        hor_std = np.sqrt(train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_bn.running_var')] + eps)
        hor_gamma = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_bn.weight')]
        hor_beta = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_bn.bias')]

        lx_kernel = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'lx_conv.weight')]
        lx_mask = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'lx_conv.mask')]
        lx_kernel = lx_kernel * lx_mask
        lx_mean = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'lx_bn.running_mean')]
        lx_std = np.sqrt(train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'lx_bn.running_var')] + eps)
        lx_gamma = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'lx_bn.weight')]
        lx_beta = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'lx_bn.bias')]

        rx_kernel = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'rx_conv.weight')]
        rx_mask = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'rx_conv.mask')]
        rx_kernel = rx_kernel * rx_mask
        rx_mean = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'rx_bn.running_mean')]
        rx_std = np.sqrt(train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'rx_bn.running_var')] + eps)
        rx_gamma = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'rx_bn.weight')]
        rx_beta = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'rx_bn.bias')]

        fused_bias = square_beta + ver_beta + hor_beta + lx_beta + rx_beta
        fused_bias -= square_mean * square_gamma / square_std
        fused_bias -= ver_mean * ver_gamma / ver_std + hor_mean * hor_gamma / hor_std
        fused_bias -= lx_mean * lx_gamma / lx_std + rx_mean * rx_gamma / rx_std

        fused_kernel = _fuse_kernel(square_kernel, square_gamma, square_std)
        _add_to_square_kernel(fused_kernel, _fuse_kernel(ver_kernel, ver_gamma, ver_std))
        _add_to_square_kernel(fused_kernel, _fuse_kernel(hor_kernel, hor_gamma, hor_std))
        _add_to_square_kernel(fused_kernel, _fuse_kernel(lx_kernel, lx_gamma, lx_std))
        _add_to_square_kernel(fused_kernel, _fuse_kernel(rx_kernel, rx_gamma, rx_std))

        deploy_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'fused_conv.weight')] = fused_kernel
        deploy_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'fused_conv.bias')] = fused_bias

    for k, v in train_dict.items():
        if 'hor_' not in k and 'ver_' not in k and 'square_' not in k and "rx_" not in k and "lx_" not in k:
            deploy_dict[k] = v
    save_hdf5(deploy_dict, deploy_weights)