# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class PReLU(nn.PReLU):
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__(num_parameters=num_parameters, init=init)
        self.init = init

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.init)

class Tanh(nn.Module):
    def __init__(self,lamda=1):
        super(Tanh, self).__init__()
        self.lamda = lamda

    def forward(self, x):
        out_forward = torch.sign(x)
        out1 = F.tanh(x*self.lamda)
        out = out_forward.detach() - out1.detach() + out1
        return out


class Leakyclip(nn.Module):
    def __init__(self):
        super(Leakyclip, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        out = x
        mask1 = x < -1
        mask2 = x > 1
        out1 = (-0.1 * x - 0.9) * mask1.type(torch.float32) + x * (1 - mask1.type(torch.float32))
        out2 = (0.1 * out1 + 0.9) * mask2.type(torch.float32) + out1 * (1 - mask2.type(torch.float32))
        out = out_forward.detach() - out2.detach() + out2

        return out

class Clip(nn.Module):
    def __init__(self):
        super(Clip, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        out = x
        mask1 = x < -1
        mask2 = x > 1
        out1 = (-1) * mask1.type(torch.float32) + x * (1 - mask1.type(torch.float32))
        out2 = (1) * mask2.type(torch.float32) + out1 * (1 - mask2.type(torch.float32))
        out = out_forward.detach() - out2.detach() + out2

        return out

class _STEQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in_):
        ctx.save_for_backward(in_)

        x = torch.sign(in_)

        return x

    @staticmethod
    def backward(ctx, grad_out_):
        (in_,) = ctx.saved_tensors

        cond = in_.abs() <= 1
        zeros = torch.zeros_like(grad_out_)
        x = torch.where(cond, grad_out_, zeros)

        return x

class STEQuantizer(torch.nn.Module):
    def forward(self, x):
        return _STEQuantizer.apply(x)

class _NORMQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in_):
        ctx.save_for_backward(in_)

        x = torch.sign(in_)
        s = in_.size()
        if len(s)==2:
            alpha = in_.norm(1, 1, keepdim=True).div(s[1]).expand(s)
        else:    
            alpha1 = in_.norm(1, 3, keepdim=True)
            alpha = alpha1.norm(1, 2, keepdim=True).div(s[2]*s[3]).expand(s)
        #alpha = in_.norm(1, 1, keepdim=True).div(s[1]).expand(s)
        x = alpha.mul(x)

        return x

    @staticmethod
    def backward(ctx, grad_out_):
        (in_,) = ctx.saved_tensors

        cond = in_.abs() <= 1
        zeros = torch.zeros_like(grad_out_)
        x = torch.where(cond, grad_out_, zeros)

        return x

class NORMQuantizer(torch.nn.Module):
    def forward(self, x):
        return _NORMQuantizer.apply(x)



bi_quantize = _STEQuantizer.apply
norm_quantize = _NORMQuantizer.apply


class BinConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        binary_weights=False,
    ):
        super(BinConv2d, self).__init__()
        """
        An implementation of a Linear layer.

        Parameters:
        - weight: the learnable weights of the module of shape (in_channels, out_channels).
        - bias: the learnable bias of the module of shape (out_channels).
        """
        self.in_channel = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.binary_weights = binary_weights

        if isinstance(kernel_size, int):
            k1, k2 = kernel_size, kernel_size
        else:
            k1, k2 = kernel_size

        self.weights_real = nn.Parameter(
            torch.Tensor(out_channels, in_channels, k1 , k2)
        )
        nn.init.kaiming_normal_(self.weights_real)

    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, *, H) where * means any number of additional
        dimensions and H = in_channels
        Output:
        - out: Output data of shape (N, *, H') where * means any number of additional
        dimensions and H' = out_channels
        """
        if self.binary_weights == 'bi':
            self.weights_real.data = torch.clamp(self.weights_real.data, -1, 1)
            weights = bi_quantize(self.weights_real)
        elif self.binary_weights == '1-norm':
            self.weights_real.data = torch.clamp(self.weights_real.data, -1, 1)
            weights = norm_quantize(self.weights_real)
        else:
            weights = self.weights_real


        out = F.conv2d(
            x,
            weights,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        return out

class BinLinear(nn.Module):
    def __init__(self, in_channels, out_channels, binary_weights=False, bias=False):
        super(BinLinear, self).__init__()
        """
        An implementation of a Linear layer.

        Parameters:
        - weight: the learnable weights of the module of shape (in_channels, out_channels).
        - bias: the learnable bias of the module of shape (out_channels).
        """
        self.in_channel = in_channels
        self.out_channels = out_channels
        self.binary_weights = binary_weights
    
        self.weights_real = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights_real)

    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, *, H) where * means any number of additional
        dimensions and H = in_channels
        Output:
        - out: Output data of shape (N, *, H') where * means any number of additional
        dimensions and H' = out_channels
        """ 
        if self.binary_weights == 'bi':
            self.weights_real.data = torch.clamp(self.weights_real.data, -1, 1)
            weights = bi_quantize(self.weights_real)
        elif self.binary_weights == '1-norm':
            self.weights_real.data = torch.clamp(self.weights_real.data, -1, 1)
            weights = norm_quantize(self.weights_real)
        else:
            weights = self.weights_real

        out = F.linear(x, weights)

        return out



class LearnedRescaleLayer2d(nn.Module):
    def __init__(self, input_shapes):
        super(LearnedRescaleLayer2d, self).__init__()
        """Implements the learned activation rescaling XNOR-Net++ style.
            This is used to scale the outputs of the binary convolutions in the Strong
            Baseline networks. [(Bulat & Tzimiropoulos,
            2019)](https://arxiv.org/abs/1909.13863)
        """

        self.shapes = input_shapes
        self.scale_a = nn.Parameter(torch.Tensor(self.shapes[1], 1, 1).fill_(1))
        self.scale_b = nn.Parameter(torch.Tensor(1, self.shapes[2], 1).fill_(1))
        self.scale_c = nn.Parameter(torch.Tensor(1, 1, self.shapes[3]).fill_(1))

    def reset_parameters(self):
        nn.init.ones_(self.scale_a)
        nn.init.ones_(self.scale_b)
        nn.init.ones_(self.scale_c)

    def forward(self, x):
        out = x * self.scale_a * self.scale_b * self.scale_c

        return out

class LearnedRescaleLayer0d(nn.Module):
    def __init__(self, input_shapes):
        super(LearnedRescaleLayer0d, self).__init__()
        """Implements the learned activation rescaling XNOR-Net++ style.
            This is used to scale the outputs of the binary convolutions in the Strong
            Baseline networks. [(Bulat & Tzimiropoulos,
            2019)](https://arxiv.org/abs/1909.13863)
        """

        self.shapes = input_shapes #1*512
        self.scale_a = nn.Parameter(torch.Tensor(self.shapes[1],).fill_(1))

    def reset_parameters(self):
        nn.init.ones_(self.scale_a)

    def forward(self, x):
        out = x * self.scale_a

        return out