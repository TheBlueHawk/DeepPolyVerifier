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
from resnet import BasicBlock


from abstract_shape import (
    AbstractShape,
    ConvAbstractShape,
    ReluAbstractShape,
    LinearAbstractShape,
    ConvAbstractShape,
    FlattenAbstractShape,
    create_abstract_input_shape,
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

    def forward(self, x: AbstractShape) -> LinearAbstractShape:
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

    def forward(self, x: AbstractShape) -> FlattenAbstractShape:
        return FlattenAbstractShape(
            y_greater=None,
            y_less=None,
            lower=x.lower.flatten(),
            upper=x.upper.flatten(),
            original_shape=x.lower.shape,
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
        self.mean = layer.mean.squeeze(0)
        self.sigma = layer.sigma.squeeze(0)

    def forward(self, x: AbstractShape) -> AbstractShape:
        # y_greater_one_neur = torch.tensor([-self.mean / self.sigma, 1 / self.sigma])
        y_greater = torch.zeros(
            *x.y_greater.shape[:-1], 1
        )  # y_greater_one_neur.repeat((x.lower.shape[0], 1))

        # y_less_one_neur = torch.tensor([-self.mean / self.sigma, 1 / self.sigma])
        y_less = torch.zeros(
            *x.y_greater.shape[:-1], 1
        )  # y_less_one_neur.repeat((x.lower.shape[0], 1))

        lower = (x.lower - self.mean) / self.sigma
        upper = (x.upper - self.mean) / self.sigma

        return AbstractShape(
            y_greater=y_greater,
            y_less=y_less,
            upper=upper,
            lower=lower,
        )


class AbstractReLU:
    def __init__(self, alpha_init="rand") -> None:
        self.alphas: Tensor = None  # Updated during forward pass
        self.alpha_init = alpha_init

    def forward(self, x: AbstractShape) -> ReluAbstractShape:
        if isinstance(x, LinearAbstractShape):
            return self.flat_forward(
                x.lower,
                x.upper,
            )
        elif isinstance(x, ConvAbstractShape):
            lower = x.lower.flatten()
            upper = x.upper.flatten()
            flat_ashape = self.flat_forward(lower, upper)
            return ReluAbstractShape(
                flat_ashape.y_greater.reshape(*x.upper.shape, 1),
                flat_ashape.y_less.reshape(*x.upper.shape, 2),
                flat_ashape.lower.reshape(*x.upper.shape),
                flat_ashape.upper.reshape(*x.upper.shape),
            )
        else:
            raise Exception("unsupported input type for relu abstract transformer")

    def flat_forward(self, l_i, u_i) -> ReluAbstractShape:
        """
        Given u_i.shape = [1,n] output AbstrcatLayer shapes:

        u_j.shape = l_j.shape = [n]
        a_greater_j = [n, 1] (list of alphas, now all alphas = 0)
        a_less_j = [n, 2] (list of linear coeff, [b,a]: b + ax)
        """

        zero_ones = torch.stack((torch.zeros_like(u_i), torch.ones_like(u_i)), dim=0)
        zero_zeros = torch.zeros_like(zero_ones)
        ones = torch.ones_like(u_i)
        zeros = torch.zeros_like(u_i)
        if self.alphas is None:
            if self.alpha_init == "rand":
                self.alphas = torch.rand_like(
                    u_i, requires_grad=True
                )  # .requires_grad_()
            elif self.alpha_init == "zeros":
                self.alphas = torch.zeros_like(u_i, requires_grad=True)
            else:
                raise Exception("Alpha intialization type not recognized")

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
        crossing = (l_i < 0) & (u_i > 0)
        slope = u_i / (u_i - l_i)
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
    def __init__(self, *args) -> None:
        if isinstance(args[0], nn.Conv2d):
            self._init_from_layer(*args)

        elif isinstance(args[0], torch.Tensor):
            self._init_from_tensor(*args)

        else:
            raise Exception(
                "Invalid arguments passed to the initializer of AbstractLinear"
            )

    def _init_from_tensor(
        self,
        kernel,
        bias,
        stride,
        padding,
        dilation=1,
    ):
        self.kernel: Tensor = kernel  # [c_out, c_in, k_h, k_w]
        self.bias: Tensor = bias  # [c_out]
        assert self.bias.dim() == 1
        assert self.kernel.shape[3] == self.kernel.shape[2]
        self.k: int = self.kernel.shape[2]
        self.c_in: int = self.kernel.shape[1]
        self.c_out: int = self.kernel.shape[0]
        self.stride: int = stride
        self.padding: int = padding
        self.dilation: int = dilation
        self.N = None

    def _init_from_layer(self, convLayer):
        self.kernel: Tensor = convLayer.weight.data  # [c_out, k_h, k_w, c_in]
        assert self.kernel.shape[3] == self.kernel.shape[2]
        self.k = self.kernel.shape[3]
        assert convLayer.kernel_size == (self.k, self.k)
        self.c_in: int = self.kernel.shape[1]
        self.c_out: int = self.kernel.shape[0]
        assert convLayer.stride[0] == convLayer.stride[1]
        self.stride: int = convLayer.stride[0]
        assert convLayer.padding[0] == convLayer.padding[1]
        self.padding: int = convLayer.padding[0]
        assert convLayer.dilation[0] == convLayer.dilation[1]
        self.dilation: int = convLayer.dilation[0]
        if convLayer.bias is None:
            self.bias = Tensor([0] * self.c_out)  # [c_out]
        else:
            self.bias = convLayer.bias.data  # [c_out]
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

        n_in = x.y_greater.shape[1]
        # n_in = 0
        self.N = conv_output_shape(
            tuple(x.lower.shape[1:]), (self.k, self.k), self.stride, self.padding
        )[0]

        x_greater = x.y_greater
        x_less = x.y_less
        x_l = x.lower.unsqueeze(0)
        x_u = x.upper.unsqueeze(0)

        w_in = x_l.shape[-1]
        h_in = x_l.shape[-2]
        h_out = math.floor(
            (h_in + 2 * self.padding - self.dilation * (self.k - 1) - 1) / self.stride
            + 1
        )
        w_out = math.floor(
            (w_in + 2 * self.padding - self.dilation * (self.k - 1) - 1) / self.stride
            + 1
        )

        l_unfold = F.unfold(
            x_l,
            kernel_size=(self.k, self.k),
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )  # [1, self.w * self.h * self.c_in , n_possible_kernel_positions]
        u_unfold = F.unfold(
            x_u,
            kernel_size=(self.k, self.k),
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

        expanded_bias = self.bias.reshape(-1, 1, 1).repeat(
            1,
            self.N,
            self.N,
        )  # [c_out, N, N]
        new_l += expanded_bias
        new_u += expanded_bias

        y_greater_one_neuron = (
            torch.concat(
                [self.bias.unsqueeze(1), self.kernel.flatten(start_dim=1)], axis=1
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )
        y_greater = y_greater_one_neuron.repeat(1, self.N, self.N, 1)

        y_less = y_greater.clone()

        return ConvAbstractShape(
            y_greater,
            y_less,
            new_l,
            new_u,
            self.c_in,
            n_in,
            self.k,
            self.padding,
            self.stride,
        )

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


class AbstractResidualSum:
    def __init__(self, *args) -> None:
        pass

    # TODO: probably change a,b to ConvAS
    def forward(self, a: AbstractShape, b: AbstractShape) -> AbstractShape:
        # TODO: need of 4D tensor <batch_size, channels, height, width>: unsqueeze?
        lower = a.lower + b.lower
        uppper = a.upper + b.upper
        # TODO: for backsub will need to compute also y_greater and y_less
        return AbstractShape(None, None, lower, uppper)


class AbstractBatchNorm:
    def __init__(self, *args) -> None:
        if isinstance(args[0], nn.BatchNorm2d):
            self._init_from_layer(*args)

        elif isinstance(args[0], torch.Tensor):
            raise NotImplementedError
            # self._init_from_tensor(*args)
        else:
            raise Exception(
                "Invalid arguments passed to the initializer of AbstractLinear"
            )

    def _init_from_layer(self, layer: nn.BatchNorm2d):
        self.num_features = layer.num_features
        self.running_mean = layer.running_mean
        self.running_var = layer.running_var
        # TODO: weights (gamma, beta) seems to be set to default, no need to retrieve them

    def _init_from_values(*args):
        pass

    def forward(self, x: ConvAbstractShape) -> AbstractShape:
        # TODO: need of 4D tensor <batch_size, channels, height, width>: unsqueeze?
        lower = F.batch_norm(x.lower, self.running_mean, self.running_var)
        uppper = F.batch_norm(x.upper, self.running_mean, self.running_var)
        # TODO: this is very likely wrong
        y_greater = F.batch_norm(x.y_greater, self.running_mean, self.running_var)
        y_less = F.batch_norm(x.y_less, self.running_mean, self.running_var)
        return ConvAbstractShape(y_greater, y_less, lower, uppper)


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

    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

    return h, w


def main():
    layer = torch.nn.Conv2d(1, 16, (4, 4), 2, 1)
    c_out, c_in, h, w = 2, 2, 3, 3
    kh, kw = 2, 2
    padding = (0, 0)
    stride = (1, 1)
    kernel = torch.concat(
        [
            torch.arange(-5, 3).reshape(1, 2, 2, 2),
            torch.arange(-2, 6).reshape(1, 2, 2, 2),
        ],
        axis=0,
    )
    bias = torch.tensor([2.0, 1])

    img = torch.zeros(c_in, h, w)
    img[:, 1, 1] += 1
    img[0, 0, :] += 1
    img[1, 1, :] += 1
    a_transformer = AbstractConvolution(kernel, bias, stride, padding)
    a_shape = create_abstract_input_shape(img, 1, bounds=(-10, 10))

    out = a_transformer.forward(a_shape)

    print("img", img, sep="\n")
    print("a_shape.upper", a_shape.upper, sep="\n")
    print("a_shape.lower", a_shape.lower, sep="\n")
    print("kernel", kernel, sep="\n")
    print("bias", bias, sep="\n")
    print("upper", out.upper, sep="\n")
    print("lower", out.lower, sep="\n")

    assert torch.allclose(
        out.upper,
        torch.tensor(
            [
                [[14.0, 12], [15, 13]],
                [[31, 29], [26, 24]],
            ]
        ),
    )

    assert torch.allclose(
        out.lower,
        torch.tensor(
            [
                [[-22.0, -24], [-21, -23]],
                [[-5, -7], [-10, -12]],
            ]
        ),
    )


if __name__ == "__main__":
    main()
