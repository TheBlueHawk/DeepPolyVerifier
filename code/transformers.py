from audioop import cross
from tokenize import Double
from turtle import forward
from typing import List, Tuple
from xmlrpc.client import Boolean
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math


from abstract_shape import (
    AbstractShape,
    ConvAbstractShape,
    ReluAbstractShape,
    LinearAbstractShape,
    ConvAbstractShape,
    Relu2DAbstractShape,
    create_abstract_input_shape
)


class AbstractLinear:
    def __init__(self, *args) -> None:
        if isinstance(args[0], nn.Linear):
            self._init_from_layer(args[0])

        elif isinstance(args[0], torch.Tensor):
            self._init_from_tensor(args[0])

        else:
            raise Exception(
                "Invalid arguments passed to the initializer of AbstractLinear"
            )

    def _init_from_tensor(self, weights):
        self.weights = weights

    def _init_from_layer(self, layer):
        self.weights = torch.cat(
            [layer.bias.data.unsqueeze(1), layer.weight.data], axis=1
        )

    def forward(self, x: AbstractShape):
        l = torch.cat([torch.tensor([1]), x.lower])
        L = l.repeat((self.weights.shape[0], 1))
        u = torch.cat([torch.tensor([1]), x.upper])
        U = u.repeat((self.weights.shape[0], 1))

        positive_weights = self.weights > 0
        LU_minor = torch.where(positive_weights, L, U)
        LU_greater = torch.where(positive_weights, U, L)

        return LinearAbstractShape(
            y_greater=self.weights,
            y_less=self.weights,
            upper=torch.sum(LU_greater * self.weights, dim=1),
            lower=torch.sum(LU_minor * self.weights, dim=1),
        )


class AbstractFlatten:
    """Produces a compact version."""

    def forward(self, x: AbstractShape):
        return AbstractShape(
            None,
            None,
            x.lower.flatten(),
            x.upper.flatten(),
        )


class AbstractNormalize:
    def __init__(self, *args) -> None:
        if len(args) == 2:
            self._init_from_values(args[0], args[1])
        elif len(args) == 1:
            self._init_from_layer(args[0])

    def _init_from_values(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def _init_from_layer(self, layer):
        self.mean = layer.mean
        self.sigma = layer.sigma

    def forward(self, x: AbstractShape):
        # y_greater_one_neur = torch.tensor([-self.mean / self.sigma, 1 / self.sigma])
        y_greater = None  # y_greater_one_neur.repeat((x.lower.shape[0], 1))

        # y_less_one_neur = torch.tensor([-self.mean / self.sigma, 1 / self.sigma])
        y_less = None  # y_less_one_neur.repeat((x.lower.shape[0], 1))

        lower = (x.lower - self.mean) / self.sigma
        upper = (x.upper - self.mean) / self.sigma

        return AbstractShape(
            y_greater=y_greater,
            y_less=y_less,
            upper=upper,
            lower=lower,
        )


class AbstractReLU:
    def __init__(self, alpha_init='rand') -> None:
        self.alphas: Tensor = None  # Updated during forward pass
        self.alpha_init = alpha_init

    def forward(self, x: AbstractShape):
        if isinstance(x, LinearAbstractShape):
            return self.flat_forward(x.upper, x.lower)
        elif isinstance(x, ConvAbstractShape):
            lower = x.lower.flatten()
            upper = x.upper.flatten()
            flat_ashape = self.flat_forward(lower, upper)
            return Relu2DAbstractShape(
                flat_ashape.y_greater.reshape(*x.upper.shape, 1),
                flat_ashape.y_less.reshape(*x.upper.shape, 2),
                flat_ashape.lower.reshape(*x.upper.shape),
                flat_ashape.upper.reshape(*x.upper.shape),
            )
        else:
            raise Exception("unsupported input type for relu abstract transformer")

    def flat_forward(self, u_i, l_i):
        # Given u_i.shape = [1,n]
        # output AbstrcatLayer shapes:
        #
        # u_j.shape = l_j.shape = [n]
        # a_greater_j = [n, 1] (list of alphas, now all alphas = 0)
        # a_less_j = [n, 2] (list of linear coeff, [b,a]: b + ax)

        zero_ones = torch.stack((torch.zeros_like(u_i), torch.ones_like(u_i)), dim=0)
        # zero_ones = torch.stack((torch.zeros_like(u_i), torch.ones_like(u_i)), dim=0)
        zero_zeros = torch.zeros_like(zero_ones)
        ones = torch.ones_like(u_i)
        zeros = torch.zeros_like(u_i)
        if self.alphas is None:
            if self.alpha_init == 'rand':
                self.alphas = torch.rand_like(u_i, requires_grad=True)  # .requires_grad_()
            elif self.alpha_init == 'zeros':
                self.alphas = torch.zeros_like(u_i, requires_grad=True)
            else:
                raise Exception("Alpha intialization type not recognized")    

        # else:
        #     self._clip_alphas()
        #     self.alphas#.requires_grad_()
        # strictly negative: zero out
        stricly_negative = u_i <= 0
        u_j = torch.where(stricly_negative, zeros, u_i)
        l_j = torch.where(stricly_negative, zeros, l_i)
        a_less_j = torch.where(stricly_negative, zero_zeros, zero_zeros)
        a_greater_j = torch.where(stricly_negative, zeros, zeros)

        # strictly positive: unchanged l,u; set y=x
        stricly_positive = l_i >= 0
        u_j = torch.where(stricly_positive, u_i, u_j)
        l_j = torch.where(stricly_positive, l_i, l_j)
        a_less_j = torch.where(stricly_positive, zero_ones, a_less_j)
        a_greater_j = torch.where(stricly_positive, ones, a_greater_j)

        # crossing: keep upperbound, lowerbound at zero, greater_than zero, less than slope
        crossing = (l_i <= 0) & (u_i >= 0)
        slope = u_i / (u_i - l_i)
        # print("slope", u_i, l_i, slope)
        u_j = torch.where(crossing, u_i, u_j)
        l_j = torch.where(crossing, torch.zeros_like(l_i), l_j)
        lin_constr = torch.stack((-1 * l_i, ones), dim=0)
        lin_constr = slope * lin_constr
        a_less_j = torch.where(crossing, lin_constr, a_less_j).T
        a_greater_j = torch.where(crossing, self.alphas * ones, a_greater_j).unsqueeze(
            1
        )

        return ReluAbstractShape(a_greater_j, a_less_j, l_j, u_j)

    def _clip_alphas(self):
        for v in self.alphas.values():
            v.data = torch.clamp(v.data, 0.0, 1.0)


class AbstractConvolution:
    def __init__(self, convLayer: torch.nn.Conv2d) -> None:
        self.kernel: Tensor = convLayer.weight.data  # [c_out, c_in, k_h, k_w]
        self.bias = convLayer.bias.data
        self.k_w: int = self.kernel.shape[3]
        self.k_h: int = self.kernel.shape[2]
        assert convLayer.kernel_size == (self.k_h, self.k_w)
        self.c_in: int = self.kernel.shape[1]
        self.c_out: int = self.kernel.shape[0]
        self.stride: tuple(int, int) = convLayer.stride
        self.padding: tuple(int, int) = convLayer.padding
        self.dilation: tuple(int, int) = convLayer.dilation
        self.N = None

    def forward(
        self,
        x: AbstractShape,
    ) -> ConvAbstractShape:
        # y_greater: tensor of shape <C, N, N, 1 + C1 * Kh * Kw>
        # y_less: tensor of shape <C, N, N, 1 + C1 * Kh * Kw>
        # lower: tensor of shape <C, N, N>
        # upper: tensor if shape <C, N N>

        # assert x.y_greater.dim == 3
        self.N = conv_output_shape(
                tuple(x.lower.shape[1:]), (self.k_h, self.k_w), self.stride,
                self.padding)[0]

        x_greater = x.y_greater
        x_less = x.y_less
        x_l = x.lower.unsqueeze(0)
        x_u = x.upper.unsqueeze(0)

        w_in = x_l.shape[-1]
        h_in = x_l.shape[-2]
        h_out = math.floor(
            (h_in + 2 * self.padding[0] - self.dilation[0] * (self.k_h - 1) - 1)
            / self.stride[0]
            + 1
        )
        w_out = math.floor(
            (w_in + 2 * self.padding[1] - self.dilation[1] * (self.k_w - 1) - 1)
            / self.stride[1]
            + 1
        )
        n_possible_kernel_positions = h_out * w_out

        l_unfold = F.unfold(
            x_l,
            kernel_size=(self.k_h, self.k_w),
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )  # [1, self.w * self.h * self.c_in , n_possible_kernel_positions]
        u_unfold = F.unfold(
            x_u,
            kernel_size=(self.k_h, self.k_w),
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )  # [1, self.w * self.h * self.c_in , n_possible_kernel_positions]
        k_flat = self.kernel.flatten(1, -1).unsqueeze(
            1
        )  # [self.c_out, 1, self.w * self.h * self.c_in]

        l_cube = torch.where(
            k_flat > 0, l_unfold.swapdims(-1, -2), u_unfold.swapdims(-1, -2)
        )  # [self.c_out, n_possible_kernel_positions, self.w * self.h * self.c_in]
        u_cube = torch.where(
            k_flat > 0, u_unfold.swapdims(-1, -2), l_unfold.swapdims(-1, -2)
        )  # [self.c_out, n_possible_kernel_positions, self.w * self.h * self.c_in]

        new_l = torch.sum(l_cube * k_flat, dim=-1).view(
            self.c_out, h_out, w_out
        )  # [self.c_out, h_out, w_out]
        new_u = torch.sum(u_cube * k_flat, dim=-1).view(
            self.c_out, h_out, w_out
        )  # [self.c_out, h_out, w_out]

        # y_greater = torch.empty((self.c_out, self.N, self.N,
        #                         1 + self.c_in * self.k_h * self.k_w))
        y_greater_one_neuron = torch.concat([self.bias.unsqueeze(1), 
                self.kernel.flatten(start_dim=1)], axis=1).unsqueeze(1).unsqueeze(1)
        y_greater = y_greater_one_neuron.repeat(1, self.N, self.N, 1)

        y_less = y_greater.clone()
        
        return ConvAbstractShape(y_greater, y_less, new_l, new_u)

    def _compute_new_l_i(self, x_l_i, x_u_i):
        x_l_i  # shape: (C_in,self.h,self.w)
        x_u_i  # shape: (C_in,self.h,self.w)
        y_l_mixed_bounds = torch.where(
            self.kernel > 0, x_l_i, x_u_i
        )  # shape: (C_in,self.h,self.w)
        conv = y_l_mixed_bounds * self.kernel  # shape: (C_out,C_in,self.h,self.w)
        l_i = torch.sum(
            conv.reshape(self.c_out, -1)
        )  # shape: (C_out,C_in * self.h * self.w)
        return l_i


# # To be used for backsubstitution
# class AbstractReLU:
#     def __init__(self) -> None:
#         pass

#     def forward(self, x: AbstractShape):
#         u_i = x.upper
#         l_i = x.lower
#         a_less_i = x.y_less
#         a_greater_i = x.y_greater
#         # strictly negative: zero out
#         stricly_negative = u_i <= 0
#         u_j = torch.where(stricly_negative, torch.zeros_like(u_i), u_i)
#         l_j = torch.where(stricly_negative, torch.zeros_like(l_i), l_i)
#         a_less_j = torch.where(stricly_negative, torch.zeros_like(u_i), a_less_i)
#         a_greater_j = torch.where(stricly_negative, torch.zeros_like(u_i), a_greater_i)

#         # strictly positive: return unchanged
#         stricly_positive = l_i >= 0
#         u_j = torch.where(stricly_positive, u_i, u_j)
#         l_j = torch.where(stricly_positive, l_i, l_j)
#         a_less_j = torch.where(stricly_positive, a_less_i, a_less_j)
#         a_greater_j = torch.where(stricly_positive, a_greater_i, a_greater_j)

#         # crossing: keep upperbound, lowerbound at zero, greater_than zero, less than slope
#         crossing = (l_i <= 0) & (u_i >= 0)
#         slope = u_i / (u_i - l_i)
#         u_j = torch.where(crossing, u_i, u_j)
#         l_j = torch.where(crossing, torch.zeros_like(l_i), l_j)
#         a_less_j = torch.where(crossing, slope * (a_greater_i - l_i), a_greater_j)
#         a_greater_j = torch.where(crossing, torch.zeros_like(u_i), a_less_j)
#         #a_greater_j = torch.where(crossing, alpha * a_greater_i, a_greater_j)

#         return AbstractShape(a_greater_j, a_less_j, u_j, l_j)


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    
    if type(stride) is not tuple:
        stride = (stride, stride)
    
    if type(pad) is not tuple:
        pad = (pad, pad)
    
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    
    return h, w

def main():
    layer = torch.nn.Conv2d(1, 16, (4,4), 2, 1)
    a_layer = AbstractConvolution(layer)
    img = torch.ones((1, 28, 28))
    a_shape = create_abstract_input_shape(img, 0.1)

    out = a_layer.forward(a_shape)

    assert out.y_greater.shape == (16, 14, 14, 17)
    assert out.y_less.shape == (16, 14, 14, 17)
    assert out.upper.shape == (16, 14, 14)
    assert out.lower.shape == (16, 14, 14)

if __name__ == "__main__":
    main()
