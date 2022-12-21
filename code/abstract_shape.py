from __future__ import annotations
from tokenize import Double
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from math import sqrt
from torch.nn.functional import conv_transpose2d


class AbstractShape:
    """Base class of all abstract shapes

    Attributes:
        y_greater: x_i >= constraint from the lecture
        y_less:  x_i <= constraint from the lecture
        lower: l from the lecture
        upper: u from the lecture
    """

    def __init__(
        self,
        y_greater: Tensor,
        y_less: Tensor,
        lower: Tensor,
        upper: Tensor,
    ) -> None:
        self.y_greater = y_greater
        self.y_less = y_less
        self.upper = upper
        self.lower = lower

    def backsub(self, previous_abstract_shape):
        pass

    def __str__(self):
        return f"y_greater\n{self.y_greater}\ny_less\n{self.y_less}\nlower\n{self.lower}\nupper\n{self.upper}"


class AbstractInputShape(AbstractShape):
    def pad(self):
        pass


class LinearAbstractShape(AbstractShape):
    """The standard representation of the output of a linear abstract layer.

    y_greater and y_less store a value for each pair of neurons in the current
    layer and the previous layer + bias.

    Attributes:
        y_greater: tensor of shape <no. curr neurons, no. prev neurons + 1>
        y_less: tensor of shape <no. curr neurons,  no. prev neurons + 1>
        lower: tensor of shape <no. curr neurons>
        upper: tensor if shape <no. curr neurons>
    """

    def backsub(self, previous_abstract_shape):
        if isinstance(previous_abstract_shape, LinearAbstractShape):
            return self.backsub_linear(previous_abstract_shape)
        elif isinstance(previous_abstract_shape, FlattenAbstractShape):
            return self.backsub_flatten(previous_abstract_shape)
        else:
            raise TypeError(type(previous_abstract_shape))

    def backsub_linear(self, previous_abstract_shape):
        greater_backsub_cube = buildConstraints3DMatrix(
            self.y_greater,
            previous_abstract_shape.y_greater,
            previous_abstract_shape.y_less,
        )

        bias_greater = self.y_greater[:, 0]
        weights_greater = self.y_greater[:, 1:].unsqueeze(1)
        new_greater = (weights_greater @ greater_backsub_cube).squeeze()
        # Add existing bias to new bias
        new_greater[:, 0] += bias_greater

        less_backsub_cube = buildConstraints3DMatrix(
            self.y_less,
            previous_abstract_shape.y_less,
            previous_abstract_shape.y_greater,
        )

        bias_less = self.y_less[:, 0]
        weights_less = self.y_less[:, 1:].unsqueeze(1)
        new_less = (weights_less @ less_backsub_cube).squeeze()
        # Add existing bias to new bias
        new_less[:, 0] += bias_less

        return LinearAbstractShape(new_greater, new_less, None, None)

    def backsub_flatten(self, flatten_ashape):
        N, N12C1 = self.y_greater.shape[0], self.y_greater.shape[1]
        return ConvAbstractShape(
            self.y_greater.reshape(N, 1, 1, N12C1),
            self.y_less.reshape(N, 1, 1, N12C1),
            None,
            None,
            c_in=flatten_ashape.original_shape[0],
            n_in=flatten_ashape.original_shape[1],
            k=flatten_ashape.original_shape[1],
            padding=0,
            stride=2,
        )


class ReluAbstractShape(AbstractShape):
    """The compact representation of the output of the abstract ReLU layer.

    y_greater stores only the slope of the line, as bias is always 0.
    y_less stores only the slope and bias of the line.

    Attributes:
        y_greater: tensor of shape <no. curr neurons, 1>
        y_less: tensor of shape <no. curr neurons, 2>
        lower: tensor of shape <no. curr neurons>
        upper: tensor if shape <<no. curr neurons>
    """

    def expand(self):
        """Rerepresents a compact relu abstract shape wrt. all neurons in the prevoious layer"""

        bias_greater = torch.zeros_like(self.y_greater)
        weights_greater = self.y_greater.flatten()
        weights_greater_expanded = torch.diag(weights_greater)
        y_greater = torch.cat([bias_greater, weights_greater_expanded], axis=1)

        bias_less = self.y_less[:, 0:1]
        weights_less = self.y_less[:, 1]
        weights_less_expanded = torch.diag(weights_less)
        y_less = torch.cat([bias_less, weights_less_expanded], axis=1)

        return LinearAbstractShape(
            y_greater, y_less, self.lower.clone(), self.upper.clone()
        )

    def backsub(self, previous_abstract_shape):
        expanded_self = self.expand()
        return expanded_self.backsub(previous_abstract_shape)


class ConvAbstractShape(AbstractShape):
    """The standard representation of the output of a convolutional abstract layer.

    y_greater and y_less store a value for each pair of neurons in the current
    layer and the previous layer + bias.

    Attributes:
        y_greater: tensor of shape <C, N, N, 1 + C1 * Kh * Kw>
        y_less: tensor of shape <C, N, N, 1 + C1 * Kh * Kw>
        lower: tensor of shape <C, N, N>
        upper: tensor if shape <C, N N>
        c_in: int - number of channels in the previous layer
        k: int - heigh/width of the kernel that produced this layer
        padding: int - padding of the kernel that produced this layer
        stride: int - stride of the kernel that produced this layer
    """

    def __init__(
        self,
        y_greater: Tensor,
        y_less: Tensor,
        lower: Tensor,
        upper: Tensor,
        c_in: int,
        n_in: int,
        k: int,
        padding: int,
        stride: int,
    ) -> None:
        super().__init__(y_greater, y_less, lower, upper)
        self.c_in = c_in
        self.n_in = n_in
        self.k = k
        self.padding = padding
        self.stride = stride

    def __str__(self):
        return (
            f"y_greater weights\n{self.y_greater[...,1:].reshape(*self.y_greater.shape[:-1], self.c_in, self.k, self.k)}\n"
            f"y_greater bias\n{self.y_greater[...,0].reshape(self.lower.shape)}"
            f"\ny_less\n{self.y_less}\nlower\n{self.lower}\nupper\n{self.upper}"
        )

    def build_padding_bitmask(self):
        bitmask = torch.ones(1, self.c_in, self.n_in, self.n_in)
        C = self.y_greater.shape[0]
        N = self.y_greater.shape[1]
        # p = self.padding
        # bitmask[:, p:-p, p:-p] = 1
        # Has zeros at the positions in y_greater/y_less that correspond to padding neurons
        unfolded_bitmask = F.unfold(
            input=bitmask,
            kernel_size=self.k,
            dilation=1,
            padding=self.padding,
            stride=self.stride,
        )

        # Reshaping to get the shape of y_greater/y_less
        unfolded_bitmask = (
            unfolded_bitmask.squeeze(dim=0)
            .swapaxes(0, 1)
            .reshape(N, N, -1)
            .unsqueeze(0)
        )
        # Adding bitmask for bias (=1)
        unfolded_bitmask = torch.cat(
            [torch.ones(*unfolded_bitmask.shape[:-1], 1), unfolded_bitmask], axis=3
        )

        return unfolded_bitmask

    def zero_out_padding_weights(self):
        # if self.k >= self.n_in + self.padding:
        #     return self.prune_padding_weights() 

        bitmask = self.build_padding_bitmask()
        new_y_greater = self.y_greater * bitmask
        new_y_less = self.y_less * bitmask

        return ConvAbstractShape(
            y_greater=new_y_greater,
            y_less=new_y_less,
            lower=None,  # self.lower.clone(),
            upper=None,  # self.upper.clone(),
            c_in=self.c_in,
            n_in=self.n_in,
            k=self.k,
            padding=self.padding,
            stride=self.stride,
        )

    # def prune_padding_weights(self):
    #     if self.k < self.n_in + self.padding or self.padding == 0:
    #         return self
        
    #     bitmask = self.build_padding_bitmask()
    #     bool_bitmask = bitmask.to(torch.bool).expand(self.y_greater.shape)
    #     new_c = self.y_greater.shape[0] * self.y_greater.shape[1] * self.y_greater.shape[2]
    #     new_y_greater = self.y_greater[bool_bitmask].reshape(new_c, 1, 1, -1)
    #     new_y_less = self.y_less[bool_bitmask].reshape(new_c, 1, 1, -1)

    #     return ConvAbstractShape(
    #         y_greater=new_y_greater,
    #         y_less=new_y_less,
    #         lower=None,  # self.lower.clone(),
    #         upper=None,  # self.upper.clone(),
    #         c_in=self.c_in,
    #         n_in=self.n_in,
    #         k=self.n_in,
    #         padding=0,
    #         stride=1,
    #     )

    def backsub(self, previous_abstract_shape: AbstractShape) -> ConvAbstractShape:
        if isinstance(previous_abstract_shape, ReluAbstractShape):
            return self.backsub_relu(previous_abstract_shape)
        elif isinstance(previous_abstract_shape, ConvAbstractShape):
            return self.backsub_conv(previous_abstract_shape)
        elif isinstance(previous_abstract_shape, ResidualAbstractShape):
            return self.backsub_res(previous_abstract_shape)
        else:
            raise TypeError(type(previous_abstract_shape))
    
    def backsub_relu(self, previous_relu_shape: ReluAbstractShape) -> ConvAbstractShape:
        # initialize the hyperparams
        cur_y_greater = self.y_greater
        cur_y_less = self.y_less
        prev_y_greater = previous_relu_shape.y_greater
        prev_y_less = previous_relu_shape.y_less

        C1 = prev_y_greater.shape[0]
        N1 = prev_y_greater.shape[1]
        C = cur_y_greater.shape[0]
        N = cur_y_greater.shape[1]
        K = self.k  # int(sqrt((cur_y_greater.shape[3] - 1) / C1))
        PADDING = self.padding
        STRIDE = self.stride

        def prep_prev_val(prev_vals):
            """
            prev_vals.shape: (C1, N1, N1)
            return.shape: (C*N*N, C1*K*K)
            """
            prev_vals = prev_vals.unsqueeze(0)
            prev_vals_unfolded = F.unfold(
                prev_vals, kernel_size=K, padding=PADDING, stride=STRIDE
            ).squeeze(dim=0)
            prev_vals_prepd = torch.cat(
                [prev_vals_unfolded for _ in range(C)], axis=1
            ).T
            return prev_vals_prepd

        def calculate_new_w(cur_w, prev_w_opt1, prev_w_opt2):
            # if kernel weight >= 0 multiply with alpha, else multiply with lambda
            cur_w_lin = cur_w.reshape(-1, C1 * K * K)
            weight_multiplier = torch.where(cur_w_lin >= 0, prev_w_opt1, prev_w_opt2)
            new_w = (cur_w_lin * weight_multiplier).reshape(C, N, N, -1)
            return new_w

        def calculate_new_bias(cur_bias, cur_w, prev_bias_opt1, prev_bias_opt2):
            cur_w_lin = cur_w.reshape(-1, C1 * K * K)
            bias_multiplier = torch.where(
                cur_w_lin >= 0, prev_bias_opt1, prev_bias_opt2
            )
            additional_bias = torch.sum(cur_w_lin * bias_multiplier, axis=1).reshape(
                C, N, N, 1
            )
            new_bias = cur_bias + additional_bias
            return new_bias

        prev_y_greater_bias_prepd = prep_prev_val(
            torch.zeros_like(prev_y_greater[..., 0])
        )
        prev_y_greater_w_prepd = prep_prev_val(prev_y_greater[..., 0])
        prev_y_less_bias_prepd = prep_prev_val(prev_y_less[..., 0])
        prev_y_less_w_prepd = prep_prev_val(prev_y_less[..., 1])

        new_y_greater_w = calculate_new_w(
            cur_y_greater[..., 1:], prev_y_greater_w_prepd, prev_y_less_w_prepd
        )
        new_y_less_w = calculate_new_w(
            cur_y_less[..., 1:], prev_y_less_w_prepd, prev_y_greater_w_prepd
        )

        new_y_greater_bias = calculate_new_bias(
            cur_y_greater[..., 0:1],
            cur_y_greater[..., 1:],
            prev_y_greater_bias_prepd,
            prev_y_less_bias_prepd,
        )
        new_y_less_bias = calculate_new_bias(
            cur_y_less[..., 0:1],
            cur_y_less[..., 1:],
            prev_y_less_bias_prepd,
            prev_y_greater_bias_prepd,
        )

        new_y_greater = torch.cat([new_y_greater_bias, new_y_greater_w], axis=3)
        new_y_less = torch.cat([new_y_less_bias, new_y_less_w], axis=3)

        return ConvAbstractShape(
            new_y_greater,
            new_y_less,
            None,
            None,
            c_in=self.c_in,
            n_in=self.n_in,
            k=self.k,
            padding=self.padding,
            stride=self.stride,
        )

    def backsub_conv(self, previous_conv_shape: ConvAbstractShape) -> ConvAbstractShape:
        cur_y_greater = self.y_greater  # <C, N, N, 1 + C1 * K * K>
        cur_y_less = self.y_less
        prev_y_greater = previous_conv_shape.y_greater  # <C1, N1, N1, 1 + C2 * K1 * K1>
        C = cur_y_greater.shape[0]
        C1 = prev_y_greater.shape[0]
        C2 = previous_conv_shape.c_in
        N = cur_y_greater.shape[1]
        N1 = prev_y_greater.shape[1]
        N2 = previous_conv_shape.n_in
        # we always use sqaured symmetric kernels
        K = self.k
        K1 = previous_conv_shape.k
        S = self.stride
        S1 = previous_conv_shape.stride
        P = self.padding
        P1 = previous_conv_shape.padding

        ### y_greater ###
        # Separate bias and reshape as <batchdim=C*N*N, kernel dims>
        inputs = cur_y_greater[:, :, :, 1:].reshape(
            (C * N * N, C1, K, K)
        )  # <C * N * N, C1, K, K> .         # doc: (minibatch,in_channels,iH,iW)
        # Separate bias, isolate kernel and reshape
        weights = (
            prev_y_greater[:, 0, 0, 1:]  # same kernel for all pixels
            .squeeze(dim=1)
            .squeeze(dim=1)
            .reshape((C1, C2, K1, K1))
        )  # <C1, C2, K1, K1>      # doc: (in_channels, out_channels,kH,kW)
        # ConvT the two kernels
        composed_kernel = conv_transpose2d(
            inputs, weights, stride=S1, padding=0
        )  # <C * N * N, C2, H_out, W_out> .   # doc: (bachdim,C_out,H_out,W_out)
        K2 = composed_kernel.shape[-1]
        composed_kernel = composed_kernel.reshape(
            C, N, N, -1
        )  # <C, N, N, C2 * H_out * W_out>

        # print(composed_kernel.shape)

        # Compute and concat bias
        bias_1 = cur_y_greater[:, :, :, 0]  # <C, N, N >
        bias_2_flat_cube = (
            prev_y_greater[:, 0, 0:1, 0:1].expand(C1, K, K).flatten()
        )  # <C1 * K * K>
        bias_2 = torch.matmul(inputs.flatten(-3, -1), bias_2_flat_cube).reshape(
            C, N, N
        )  # <C, N, N>
        bias = bias_1 + bias_2  # <C, N, N>
        composed_kernel_with_b_greater = torch.cat(
            (bias.unsqueeze(-1), composed_kernel), -1
        )  # <C, N, N, C2 * H_out * W_out + 1>

        ### y_less ###
        # Separate bias and reshape as <batchdim=C*N*N, kernel dims>
        inputs = cur_y_less[:, :, :, 1:].reshape(
            (C * N * N, C1, K, K)
        )  # <C * N * N, C1, K, K> .         # doc: (minibatch,in_channels,iH,iW)
        # Separate bias, isolate kernel and reshape
        # prev_y_greater = prev_y_less for a conv layer
        weights = (
            prev_y_greater[:, 0, 0, 1:]  # same kernel for all pixels
            .squeeze(dim=1)
            .squeeze(dim=1)
            .reshape((C1, C2, K1, K1))
        )  # <C1, C2, K1, K1>      # doc: (in_channels, out_channels,kH,kW)
        # ConvT the two kernels
        composed_kernel = conv_transpose2d(
            inputs, weights, stride=S1, padding=0
        )  # <C * N * N, C2, H_out, W_out> .   # doc: (bachdim,C_out,H_out,W_out)
        K2 = composed_kernel.shape[-1]
        composed_kernel = composed_kernel.reshape(
            C, N, N, -1
        )  # <C, N, N, C2 * H_out * W_out>

        # print(composed_kernel.shape)

        # Compute and concat bias
        bias_1 = cur_y_less[:, :, :, 0]  # <C, N, N >
        bias_2_flat_cube = (
            prev_y_greater[:, 0, 0:1, 0:1].expand(C1, K, K).flatten()
        )  # <C1 * K * K>
        bias_2 = torch.matmul(inputs.flatten(-3, -1), bias_2_flat_cube).reshape(
            C, N, N
        )  # <C, N, N>
        bias = bias_1 + bias_2  # <C, N, N>
        composed_kernel_with_b_less = torch.cat(
            (bias.unsqueeze(-1), composed_kernel), -1
        )  # <C, N, N, C2 * H_out * W_out + 1>

        S2 = S * S1
        P2 = P1 + S1 * P

        return ConvAbstractShape(
            composed_kernel_with_b_greater,
            composed_kernel_with_b_less,
            None,
            None,
            c_in=C2,
            n_in=N2,
            k=K2,
            padding=P2,
            stride=S2,
        )

    def backsub_res(self, previous_shape: ResidualAbstractShape) -> ConvAbstractShape:
        # Path a
        backsub_path_a = self.backsub(previous_shape.conv1_bn_a)
        backsub_path_a = backsub_path_a.zero_out_padding_weights()
        # Remove current conv bias to avoid double counting
        

        # copy current shape and zero out bias
        weights_bitmask = torch.ones_like(self.y_greater)
        weights_bitmask[...,0] = 0 # zero out bias
        curr_shape_b = ConvAbstractShape(
            y_greater=self.y_greater * weights_bitmask, 
            y_less=self.y_less * weights_bitmask, 
            lower=self.lower, 
            upper=self.upper, 
            c_in=self.c_in, 
            n_in=self.n_in, 
            k=self.k, 
            padding=self.padding, 
            stride=self.stride)
        # self.y_greater[:, :, :, 0] = 0
        # self.y_less[:, :, :, 0] = 0
        # Path b
        backsub_path_b = curr_shape_b.backsub(previous_shape.conv2_bn_b)
        backsub_path_b = backsub_path_b.zero_out_padding_weights()
        backsub_path_b = backsub_path_b.backsub(previous_shape.relu1b)
        backsub_path_b = backsub_path_b.backsub(previous_shape.conv1_bn_b)
        backsub_path_b = backsub_path_b.zero_out_padding_weights()
        merged_abstract_shape = previous_shape.merge(backsub_path_a, backsub_path_b)
        return merged_abstract_shape


class FlattenAbstractShape(AbstractShape):
    def __init__(
        self,
        y_greater: Tensor,
        y_less: Tensor,
        lower: Tensor,
        upper: Tensor,
        original_shape,
    ) -> None:
        super().__init__(y_greater=y_greater, y_less=y_less, lower=lower, upper=upper)
        self.original_shape = original_shape

    def backsub(self, previous_abstract_shape):
        pass


class ResidualAbstractShape(AbstractInputShape):
    def __init__(
        self,
        y_greater: Tensor,
        y_less: Tensor,
        lower: Tensor,
        upper: Tensor,
        conv1_bn_a: ConvAbstractShape,
        conv1_bn_b: ConvAbstractShape,
        relu1b: ReluAbstractShape,
        conv2_bn_b: ConvAbstractShape,
    ) -> None:
        super().__init__(y_greater, y_less, lower, upper)
        # Path a
        self.conv1_bn_a = conv1_bn_a
        self.conv1_bn_b = conv1_bn_b
        self.relu1b = relu1b
        self.conv2_bn_b = conv2_bn_b

    def backsub(self, previous_abstract_shape: ReluAbstractShape) -> ConvAbstractShape:
        temp_convAS = self.internal_backsub()
        abstract_shape = temp_convAS.backsub(previous_abstract_shape)
        return abstract_shape

    def internal_backsub(
        self,
    ) -> ConvAbstractShape:
        backsub_path_a = self.conv1_bn_a
        backsub_path_b = self.conv2_bn_b.backsub(self.relu1b)
        backsub_path_b = backsub_path_b.backsub(self.conv1_bn_b)
        merged_abstract_shape = self.merge(backsub_path_a, backsub_path_b)
        return merged_abstract_shape

    def merge(self, a: ConvAbstractShape, b: ConvAbstractShape) -> ConvAbstractShape:
        # Kernel in b is larger than the kernel in a
        return ConvAbstractShape(
            y_greater=self.submerge(a.y_greater, b.y_greater, a.c_in, a.k, b.k),
            y_less=self.submerge(a.y_less, b.y_less, a.c_in, a.k, b.k),
            lower=None,
            upper=None,
            c_in=b.c_in,
            n_in=b.n_in,
            k=b.k,
            padding=b.padding,
            stride=b.stride,
        )

    def submerge(self, ineq_a, ineq_b, c_in, ka, kb):
        """Merges y_greater or y_lower from 2 abstract shapes"""
        bias_a = ineq_a[..., 0:1]
        bias_b = ineq_b[..., 0:1]
        bias_comb = bias_a + bias_b

        weights_a = ineq_a[..., 1:]
        weights_b = ineq_b[..., 1:]

        weights_a = weights_a.reshape(*weights_a.shape[:-1], c_in, ka, ka)
        pad = (int)((kb - ka) / 2)
        padding = (pad, pad, pad, pad)
        weights_a_expanded = F.pad(weights_a, padding, "constant", 0)
        weights_a_expanded = weights_a_expanded.flatten(start_dim=3, end_dim=5)
        weights_comb = weights_a_expanded + weights_b

        ineq_comb = torch.concat([bias_comb, weights_comb], axis=3)
        return ineq_comb


def create_abstract_input_shape(inputs, eps, bounds=(0, 1)) -> AbstractShape:
    return AbstractInputShape(
        y_greater=torch.clamp(inputs - eps, bounds[0], bounds[1]).unsqueeze(-1),
        y_less=torch.clamp(inputs + eps, bounds[0], bounds[1]).unsqueeze(-1),
        lower=torch.clamp(inputs - eps, bounds[0], bounds[1]),
        upper=torch.clamp(inputs + eps, bounds[0], bounds[1]),
    )


def buildConstraints3DMatrix(
    cur_weights: Tensor,
    fst_choice,
    snd_choice,
) -> Tensor:
    """
    Where cur_weighs >= 0 takes fst_choice, otherwise takes snd_choice.

    E.g, when building it for cur_weights = current_ashape.upper:
        fst_choice = current_weights.upper
        snd_choice = current_weights.lower
    """
    # curr_y_greater = current_layer_ashape.y_greater  # shape: [N, N1+1]
    # prev_y_greater = previous_layer_ashape.y_greater  # shape: [N1, N2+1]
    # prev_y_smaller = previous_layer_ashape.y_less  # shape: [N1, N2+1]
    # N = curr_y_greater.shape[0]  # n_neurons_current_layer
    # N1 = curr_y_greater.shape[1]  # n_neurons_prev_layer
    # N2 = prev_y_greater.shape[1]  # n_neurons_prev_prev_layer
    assert cur_weights.shape[1] - 1 == fst_choice.shape[0] == snd_choice.shape[0]
    # curr_b = curr_y_greater[:, 0].unsqueeze(1)  # shape: [N, 1, 1]
    # bias = torch.concat((curr_b, torch.zeros(N, N2, 1)), dim=1)  # shape: [N, N2+1, 1]
    alphas = cur_weights[:, 1:].unsqueeze(1)  # shape: [N, 1, N1]
    cube = torch.where(
        alphas.swapdims(1, 2) >= 0, fst_choice, snd_choice
    )  # shape: [N, N1, N2 + 1]
    return cube


# def buildConstraints3DMatrix(
#     current_layer_ashape: AbstractShape, previous_layer_ashape: AbstractShape
# ) -> Tensor:
#     curr_y_greater = current_layer_ashape.y_greater  # shape: [N, N1+1]
#     prev_y_greater = previous_layer_ashape.y_greater  # shape: [N1, N2+1]
#     prev_y_smaller = previous_layer_ashape.y_less  # shape: [N1, N2+1]
#     # N = curr_y_greater.shape[0]  # n_neurons_current_layer
#     # N1 = curr_y_greater.shape[1]  # n_neurons_prev_layer
#     # N2 = prev_y_greater.shape[1]  # n_neurons_prev_prev_layer
#     assert (
#         curr_y_greater.shape[1] - 1
#         == prev_y_smaller.shape[0]
#         == prev_y_greater.shape[0]
#     )
#     # curr_b = curr_y_greater[:, 0].unsqueeze(1)  # shape: [N, 1, 1]
#     # bias = torch.concat((curr_b, torch.zeros(N, N2, 1)), dim=1)  # shape: [N, N2+1, 1]
#     alphas = curr_y_greater[:, 1:].unsqueeze(1)  # shape: [N, 1, N1]
#     cube = torch.where(
#         alphas.swapdims(1, 2) >= 0, prev_y_greater, prev_y_smaller
#     )  # shape: [N, N1, N2 + 1]
#     return cube


def weightedLoss(output: Tensor, gamma: Double) -> Tensor:
    """Compute the negative weighted sum of the tensors outputs.
    Negative entries in the output tensors are weighted more with gamma.
    This loss incentivize changes in paramters that cause high probabilities in non-target outputs

    Args:
        output (Tensor): the output of final layer of the abstract network: [(x_t - x_i), ...]
        gamma (Double): the weighting factor for negative entries, >= 1

    Returns:
        Double: loss
    """
    assert gamma >= 1
    weighted_out = torch.where(output < 0, output * gamma, output)
    negsum = torch.sum(weighted_out) * (-1)
    return negsum


def main():
    pass


if __name__ == "__main__":
    main()
