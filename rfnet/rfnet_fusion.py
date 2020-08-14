from utils.misc import read_hdf5, save_hdf5
import numpy as np

from custom_layers.clamp_conv1x1 import BOUND

SQUARE_KERNEL_KEYWORD = 'main_conv.weight'

def _fuse_kernel(kernel, gamma, std):
    b_gamma = np.reshape(gamma, (kernel.shape[0], 1, 1, 1))
    b_gamma = np.tile(b_gamma, (1, kernel.shape[1], kernel.shape[2], kernel.shape[3]))
    b_std = np.reshape(std, (kernel.shape[0], 1, 1, 1))
    b_std = np.tile(b_std, (1, kernel.shape[1], kernel.shape[2], kernel.shape[3]))
    return kernel * b_gamma / b_std

def _merge_kernel(curr, prev):
    # prev = np.clip(prev, -BOUND, BOUND)
    curr_t = np.transpose(curr, [2, 3, 0, 1])
    prev_t = np.transpose(prev, [2, 3, 0, 1])
    merged = np.matmul(curr_t, prev_t)
    return np.transpose(merged, [2, 3, 0, 1])

def _add_to_main_kernel(main_kernel, asym_kernel):
    '''
    asym_h = asym_kernel.shape[2]
    asym_w = asym_kernel.shape[3]
    main_h = main_kernel.shape[2]
    main_w = main_kernel.shape[3]
    main_kernel[:, :, main_h // 2 - asym_h // 2: main_h // 2 - asym_h // 2 + asym_h,
                                        main_w // 2 - asym_w // 2 : main_w // 2 - asym_w // 2 + asym_w] += asym_kernel
    '''
    main_kernel[:] += asym_kernel

def convert_rfnet_weights(train_weights, deploy_weights, eps):
    train_dict = read_hdf5(train_weights)
    print(train_dict.keys())
    deploy_dict = {}
    main_conv_var_names = [name for name in train_dict.keys() if SQUARE_KERNEL_KEYWORD in name]
    for main_name in main_conv_var_names:
        # main_prev = train_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, 'main_alter.weight')]
        main_kernel = train_dict[main_name]
        # main_kernel = _merge_kernel(main_kernel, main_prev)
        main_mean = train_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, 'main_bn.running_mean')]
        main_std = np.sqrt(train_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, 'main_bn.running_var')] + eps)
        main_gamma = train_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, 'main_bn.weight')]
        main_beta = train_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, 'main_bn.bias')]

        left_prev = train_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, "left_alter.weight")]
        left_kernel = train_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, 'left_conv.weight')]
        left_kernel = _merge_kernel(left_kernel, left_prev)
        left_mean = train_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, 'left_bn.running_mean')]
        left_std = np.sqrt(train_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, 'left_bn.running_var')] + eps)
        left_gamma = train_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, 'left_bn.weight')]
        left_beta = train_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, 'left_bn.bias')]

        right_prev = train_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, "right_alter.weight")]
        right_kernel = train_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, 'right_conv.weight')]
        right_kernel = _merge_kernel(right_kernel, right_prev)
        right_mean = train_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, 'right_bn.running_mean')]
        right_std = np.sqrt(train_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, 'right_bn.running_var')] + eps)
        right_gamma = train_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, 'right_bn.weight')]
        right_beta = train_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, 'right_bn.bias')]

        fused_bias = main_beta + left_beta + right_beta - main_mean * main_gamma / main_std \
                     - left_mean * left_gamma / left_std - right_mean * right_gamma / right_std
        fused_kernel = _fuse_kernel(main_kernel, main_gamma, main_std)
        _add_to_main_kernel(fused_kernel, _fuse_kernel(left_kernel, left_gamma, left_std))
        _add_to_main_kernel(fused_kernel, _fuse_kernel(right_kernel, right_gamma, right_std))

        deploy_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, 'fused_conv.weight')] = fused_kernel
        deploy_dict[main_name.replace(SQUARE_KERNEL_KEYWORD, 'fused_conv.bias')] = fused_bias

    for k, v in train_dict.items():
        if 'right_' not in k and 'left_' not in k and 'main_' not in k:
            deploy_dict[k] = v
    save_hdf5(deploy_dict, deploy_weights)