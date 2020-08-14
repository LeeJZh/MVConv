import torch
from torch import alpha_dropout, nn
from torch.autograd import Function, grad


class Diff_Reg_Layer(Function):
    @staticmethod
    def forward(ctx, left, right, alpha):
        ctx.save_for_backward(left, right)
        ctx.alpha = alpha
        return left, right
    
    @staticmethod
    def backward(ctx, grad_out_left, grad_out_right):
        left, right = ctx.saved_tensors
        alpha = ctx.alpha
        grad_left = grad_right = grad_alpha = None
        N = left.size(0)
        grad_left = grad_out_left - alpha * (left - right) * 2 / N
        grad_right = grad_out_right + alpha * (left - right) * 2 / N
        return grad_left, grad_right, grad_alpha


diff_reg = Diff_Reg_Layer.apply

class DiffRegLayer(nn.Module):
    def __init__(self, alpha=(1e-6)) -> None:
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, left, right):
        return diff_reg(left, right, self.alpha)
        # if self.training:
        #     diff = -self.l1(left, right) * self.alpha
        #     diff.backward(retain_graph=True)
        # return left, right