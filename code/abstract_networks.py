from abstract_shape import (
    AbstractShape,
    ConvAbstractShape,
    LinearAbstractShape,
    ReluAbstractShape,
    ResidualAbstractShape,
)
from typing import List, Tuple
from torch import Tensor
import torch
from resnet import BasicBlock, ResNet
from transformers import (
    AbstractBatchNorm,
    AbstractConvolution,
    AbstractNormalize,
    AbstractFlatten,
    AbstractLinear,
    AbstractReLU,
)
from anet_checkers import ANetChecker
from networks import NormalizedResnet


def recompute_bounds_linear(first_ashape, curr_ashape):
    tmp_ashape_u = AbstractLinear(curr_ashape.y_less).forward(first_ashape)
    tmp_ashape_l = AbstractLinear(curr_ashape.y_greater).forward(first_ashape)
    return tmp_ashape_u.upper, tmp_ashape_l.lower


def recompute_bounds_conv(first_ashape, curr_ashape):
    kernel_lower = curr_ashape.y_greater[
        :, :, :, 1:
    ]  # <C, N, N, C2 * composedK * composedK>
    kernel_lower = kernel_lower.flatten(0, 2)  # <C * N * N, C2 * composedK * composedK>
    kernel_lower = kernel_lower.reshape(
        kernel_lower.shape[0], curr_ashape.c_in, curr_ashape.k, curr_ashape.k
    )  # <C * N * N, C2, composedK, composedK> = [c_out, c_in, k,k]
    bias = curr_ashape.y_greater[:, :, :, 0].flatten()  # <C * N * N>
    temp_ashape = AbstractConvolution(
        kernel_lower,
        bias,
        curr_ashape.stride,
        curr_ashape.padding,
    ).forward(first_ashape)
    new_l = temp_ashape.lower  # <C * N * N, N, N> # ... need to select idx
    new_N = new_l.shape[-1]
    new_l = (
        new_l.reshape(-1, new_N, new_N, new_N, new_N)
        .diagonal(dim1=1, dim2=3)
        .diagonal(dim1=1, dim2=2)
    )

    kernel_upper = curr_ashape.y_less[
        :, :, :, 1:
    ]  # <C, N, N, C2 * composedK * composedK>
    kernel_upper = kernel_upper.flatten(0, 2)  # <C * N * N, C2 * composedK * composedK>
    kernel_upper = kernel_upper.reshape(
        kernel_upper.shape[0], curr_ashape.c_in, curr_ashape.k, curr_ashape.k
    )  # <C * N * N, C2, composedK, composedK> = [c_out, c_in, k,k]
    bias = curr_ashape.y_less[:, :, :, 0].flatten()  # <C * N * N>
    temp_ashape = AbstractConvolution(
        kernel_upper,
        bias,
        curr_ashape.stride,
        curr_ashape.padding,
    ).forward(first_ashape)
    new_u = temp_ashape.upper  # <C * N * N, N, N>  # overcomputation ...
    new_N = new_u.shape[-1]
    new_u = (
        new_u.reshape(-1, new_N, new_N, new_N, new_N)
        .diagonal(dim1=1, dim2=3)
        .diagonal(dim1=1, dim2=2)
    )

    return new_u, new_l


def recompute_bounds(first_ashape, curr_ashape, out_ashape):
    # Recompute l & u
    if isinstance(curr_ashape, LinearAbstractShape):
        out_ashape.upper, out_ashape.lower = recompute_bounds_linear(
            first_ashape, curr_ashape
        )

    elif isinstance(curr_ashape, ConvAbstractShape):
        out_ashape.upper, out_ashape.lower = recompute_bounds_conv(
            first_ashape, curr_ashape
        )
        if isinstance(out_ashape, LinearAbstractShape):
            out_ashape.upper = torch.squeeze(out_ashape.upper)
            out_ashape.lower = torch.squeeze(out_ashape.lower)


class AbstractNetwork:
    def __init__(
        self, abstract_transformers: List, checker: ANetChecker = None
    ) -> None:
        self.abstract_transformers = abstract_transformers
        self.checker = checker

    def backsub(
        self, abstract_shape: AbstractShape, previous_shapes, check=False
    ) -> AbstractShape:
        curr_ashape = abstract_shape
        for i, previous_shape in enumerate(reversed(previous_shapes[1:])):
            if isinstance(curr_ashape, ConvAbstractShape):
                curr_ashape = curr_ashape.zero_out_padding_weights()
            if check:
                recompute_bounds(previous_shape, curr_ashape, abstract_shape)
                self.checker.recheck(abstract_shape)
            curr_ashape = curr_ashape.backsub(previous_shape)

        if isinstance(curr_ashape, ConvAbstractShape):
            curr_ashape = curr_ashape.zero_out_padding_weights()
        recompute_bounds(previous_shapes[0], curr_ashape, abstract_shape)
        if check:
            self.checker.recheck(abstract_shape)

        return abstract_shape

    def get_abstract_transformers(self):
        return self.abstract_transformers

    def buildFinalLayerWeights(self, true_lablel: int, N: int) -> Tensor:
        weights = torch.zeros(N, N)
        for i in range(N):
            if i == true_lablel:
                weights.T[i] = torch.ones(N)
            weights[i][i] += -1

        bias = torch.zeros(N, 1)
        wb = torch.cat((bias, weights), dim=-1)
        return wb

    def buildFinalLayer(self, true_label: int, N: int) -> bool:
        final_layer: AbstractLinear = AbstractLinear(
            self.buildFinalLayerWeights(true_label, N)
        )
        return final_layer

    def get_alphas() -> List[Tensor]:
        raise NotImplementedError

    def set_alphas() -> List[Tensor]:
        raise NotImplementedError


class AbstractNet1(AbstractNetwork):
    def __init__(self, net, checker) -> None:
        self.normalize = AbstractNormalize(net.layers[0])
        self.flatten = AbstractFlatten()
        self.lin1 = AbstractLinear(net.layers[2])
        self.relu1 = AbstractReLU()
        self.lin2 = AbstractLinear(net.layers[4])
        self.final_atransformer = None  # built in forward

        self.checker = checker

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []
        self.checker.check_next(abstract_shape)

        # No need to backsub Normalization and Flatten, they operate pointwise
        abstract_shape = self.normalize.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        abstract_shape = self.flatten.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        # Won't get tighter l, u after backsubstituting the first linear layer
        abstract_shape = self.lin1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape).expand()
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)

        return abstract_shape

    def get_alphas(self) -> List[Tensor]:
        alphas = [self.relu1.real_alphas]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 1
        assert self.relu1.real_alphas.shape == updated_alphas[0].shape
        self.relu1.real_alphas = updated_alphas[0]


class AbstractNet2(AbstractNetwork):
    def __init__(self, net, checker) -> None:
        self.normalize = AbstractNormalize(net.layers[0])
        self.flatten = AbstractFlatten()
        self.lin1 = AbstractLinear(net.layers[2])
        self.relu1 = AbstractReLU()
        self.lin2 = AbstractLinear(net.layers[4])
        self.relu2 = AbstractReLU()
        self.lin3 = AbstractLinear(net.layers[6])
        self.final_atransformer = None  # built in forward

        self.checker = checker

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []
        self.checker.check_next(abstract_shape)

        # No need to backsub Normalization and Flatten, they operate pointwise
        abstract_shape = self.normalize.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        abstract_shape = self.flatten.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        # Won't get tighter l, u after backsubstituting the first linear layer
        abstract_shape = self.lin1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape).expand()
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes, check=True)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape).expand()
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin3.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)

        return abstract_shape

    def get_alphas(self) -> List[Tensor]:
        alphas = [self.relu1.real_alphas, self.relu2.real_alphas]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 2
        assert self.relu1.real_alphas.shape == updated_alphas[0].shape
        self.relu1.real_alphas = updated_alphas[0]
        assert self.relu2.real_alphas.shape == updated_alphas[1].shape
        self.relu2.real_alphas = updated_alphas[1]


class AbstractNet3(AbstractNetwork):
    def __init__(self, net, checker) -> None:
        self.normalize = AbstractNormalize(net.layers[0])
        self.flatten = AbstractFlatten()
        self.lin1 = AbstractLinear(net.layers[2])
        self.relu1 = AbstractReLU()
        self.lin2 = AbstractLinear(net.layers[4])
        self.relu2 = AbstractReLU()
        self.lin3 = AbstractLinear(net.layers[6])
        self.final_atransformer = None  # built in forward

        self.checker = checker

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []

        # No need to backsub Normalization and Flatten, they operate pointwise
        abstract_shape = self.normalize.forward(abstract_shape)
        abstract_shape = self.flatten.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        # Won't get tighter l, u after backsubstituting the first linear layer
        abstract_shape = self.lin1.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape).expand()
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape).expand()
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin3.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)

        return abstract_shape

    def get_alphas(self) -> List[Tensor]:
        alphas = [self.relu1.real_alphas, self.relu2.real_alphas]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 2
        assert self.relu1.real_alphas.shape == updated_alphas[0].shape
        self.relu1.real_alphas = updated_alphas[0]
        assert self.relu2.real_alphas.shape == updated_alphas[1].shape
        self.relu2.real_alphas = updated_alphas[1]


class AbstractNet4(AbstractNetwork):
    def __init__(self, net, checker) -> None:
        # Conv(device, "mnist", 28, 1, [(16, 3, 2, 1)], [100, 10], 10)
        self.normalize = AbstractNormalize(net.layers[0])
        self.conv1 = AbstractConvolution(net.layers[1])
        self.relu1 = AbstractReLU()
        self.flatten = AbstractFlatten()
        self.lin1 = AbstractLinear(net.layers[4])
        self.relu2 = AbstractReLU()
        self.lin2 = AbstractLinear(net.layers[6])
        self.final_atransformer = None  # built in forward

        self.checker = checker

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []
        self.checker.check_next(abstract_shape)

        # No need to backsub Normalization, it operates pointwise
        abstract_shape = self.normalize.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        # Won't get tighter l, u after backsubstituting the first conv layer
        abstract_shape = self.conv1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.flatten.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes, check=True)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape).expand()
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)

        return abstract_shape

    def get_alphas(self) -> List[Tensor]:
        alphas = [self.relu1.real_alphas, self.relu2.real_alphas]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 2
        assert self.relu1.real_alphas.shape == updated_alphas[0].shape
        self.relu1.real_alphas = updated_alphas[0]
        assert self.relu2.real_alphas.shape == updated_alphas[1].shape
        self.relu2.real_alphas = updated_alphas[1]


class AbstractNet5(AbstractNetwork):
    def __init__(self, net, checker) -> None:
        # Conv(device, "mnist", 28, 1, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10)
        self.normalize = AbstractNormalize(net.layers[0])
        self.conv1 = AbstractConvolution(net.layers[1])
        self.relu1 = AbstractReLU()
        self.conv2 = AbstractConvolution(net.layers[3])
        self.relu2 = AbstractReLU()
        self.flatten = AbstractFlatten()
        self.lin1 = AbstractLinear(net.layers[6])
        self.relu3 = AbstractReLU()
        self.lin2 = AbstractLinear(net.layers[8])
        self.final_atransformer = None  # built in forward

        self.checker = checker

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []
        self.checker.check_next(abstract_shape)

        # No need to backsub Normalization, it operates pointwise
        abstract_shape = self.normalize.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        # Won't get tighter l, u after backsubstituting the first linear layer
        abstract_shape = self.conv1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.conv2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes, check=True)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.flatten.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin1.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu3.forward(abstract_shape).expand()
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)

        return abstract_shape

    def get_alphas(self) -> List[Tensor]:
        alphas = [self.relu1.real_alphas, self.relu2.real_alphas, self.relu3.real_alphas]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 3
        assert self.relu1.real_alphas.shape == updated_alphas[0].shape
        self.relu1.real_alphas = updated_alphas[0]
        assert self.relu2.real_alphas.shape == updated_alphas[1].shape
        self.relu2.real_alphas = updated_alphas[1]
        assert self.relu3.real_alphas.shape == updated_alphas[2].shape
        self.relu3.real_alphas = updated_alphas[2]


class AbstractNet6(AbstractNetwork):
    def __init__(self, net, checker) -> None:
        # Conv(device, "cifar10", 32, 3, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10)
        self.normalize = AbstractNormalize(net.layers[0])
        self.conv1 = AbstractConvolution(net.layers[1])
        self.relu1 = AbstractReLU()
        self.conv2 = AbstractConvolution(net.layers[3])
        self.relu2 = AbstractReLU()
        self.flatten = AbstractFlatten()
        self.lin1 = AbstractLinear(net.layers[6])
        self.relu3 = AbstractReLU()
        self.lin2 = AbstractLinear(net.layers[8])
        self.final_atransformer = None  # built in forward

        self.checker = checker

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []
        self.checker.check_next(abstract_shape)

        # No need to backsub Normalization, it operates pointwise
        abstract_shape = self.normalize.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        # Won't get tighter l, u after backsubstituting the first linear layer
        abstract_shape = self.conv1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.conv2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes, check=True)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.flatten.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes, check=True)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu3.forward(abstract_shape).expand()
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)

        return abstract_shape

    def get_alphas(self) -> List[Tensor]:
        alphas = [self.relu1.real_alphas, self.relu2.real_alphas, self.relu3.real_alphas]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 3
        assert self.relu1.real_alphas.shape == updated_alphas[0].shape
        self.relu1.real_alphas = updated_alphas[0]
        assert self.relu2.real_alphas.shape == updated_alphas[1].shape
        self.relu2.real_alphas = updated_alphas[1]
        assert self.relu3.real_alphas.shape == updated_alphas[2].shape
        self.relu3.real_alphas = updated_alphas[2]


class AbstractNet7(AbstractNetwork):
    def __init__(self, net, checker) -> None:
        # Conv(device, "mnist", 28, 1, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10)
        self.normalize = AbstractNormalize(net.layers[0])
        self.conv1 = AbstractConvolution(net.layers[1])
        self.relu1 = AbstractReLU()
        self.conv2 = AbstractConvolution(net.layers[3])
        self.relu2 = AbstractReLU()
        self.flatten = AbstractFlatten()
        self.lin1 = AbstractLinear(net.layers[6])
        self.relu3 = AbstractReLU()
        self.lin2 = AbstractLinear(net.layers[8])
        self.relu4 = AbstractReLU()
        self.lin3 = AbstractLinear(net.layers[10])
        self.final_atransformer = None  # built in forward

        self.checker = checker

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []
        self.checker.check_next(abstract_shape)

        # No need to backsub Normalization, it operates pointwise
        abstract_shape = self.normalize.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        # Won't get tighter l, u after backsubstituting the first linear layer
        abstract_shape = self.conv1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.conv2.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.flatten.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin1.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu3.forward(abstract_shape).expand()
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu4.forward(abstract_shape).expand()
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin3.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)

        return abstract_shape

    def get_alphas(self) -> List[Tensor]:
        alphas = [
            self.relu1.real_alphas,
            self.relu2.real_alphas,
            self.relu3.real_alphas,
            self.relu4.real_alphas,
        ]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 4
        assert self.relu1.real_alphas.shape == updated_alphas[0].shape
        self.relu1.real_alphas = updated_alphas[0]
        assert self.relu2.real_alphas.shape == updated_alphas[1].shape
        self.relu2.real_alphas = updated_alphas[1]
        assert self.relu3.real_alphas.shape == updated_alphas[2].shape
        self.relu3.real_alphas = updated_alphas[2]
        assert self.relu4.real_alphas.shape == updated_alphas[3].shape
        self.relu4.real_alphas = updated_alphas[3]


class AbstractNet8(AbstractNetwork):
    def __init__(self, net: NormalizedResnet, checker) -> None:
        self.normalize = AbstractNormalize(net.normalization)
        resnet = net.resnet

        self.conv1 = AbstractConvolution(resnet[0])
        self.relu1 = AbstractReLU()

        blockLayers1: torch.nn.Sequential = resnet[2]
        self.block1 = AbstractBlockSubnet(blockLayers1[0], checker)
        self.relu2 = AbstractReLU()
        self.block2 = AbstractBlockSubnet(blockLayers1[2], checker)
        self.relu3 = AbstractReLU()

        self.flatten = AbstractFlatten()
        self.lin1 = AbstractLinear(resnet[4])
        self.relu4 = AbstractReLU()
        self.lin2 = AbstractLinear(resnet[6])
        self.final_atransformer = None  # built in forward

        self.checker = checker

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []
        self.checker.check_next(abstract_shape)

        abstract_shape = self.normalize.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.conv1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.block1.forward(abstract_shape, prev_abstract_shapes)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.block2.forward(abstract_shape, prev_abstract_shapes)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu3.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.flatten.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin1.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu4.forward(abstract_shape).expand()
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)

        return abstract_shape

    def get_alphas(self) -> List[Tensor]:
        alphas = [
            self.relu1.real_alphas,
            self.block1.relu1b.real_alphas,
            self.relu2.real_alphas,
            self.block2.relu1b.real_alphas,
            self.relu3.real_alphas,
            self.relu4.real_alphas,
        ]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 6
        assert self.relu1.real_alphas.shape == updated_alphas[0].shape
        self.relu1.real_alphas = updated_alphas[0]
        assert self.block1.relu1b.real_alphas.shape == updated_alphas[1].shape
        self.block1.relu1b.real_alphas = updated_alphas[1]
        assert self.relu2.real_alphas.shape == updated_alphas[2].shape
        self.relu2.real_alphas = updated_alphas[2]
        assert self.block2.relu1b.real_alphas.shape == updated_alphas[3].shape
        self.block2.relu1b.real_alphas = updated_alphas[3]
        assert self.relu3.real_alphas.shape == updated_alphas[4].shape
        self.relu3.real_alphas = updated_alphas[4]
        assert self.relu4.real_alphas.shape == updated_alphas[5].shape
        self.relu4.real_alphas = updated_alphas[5]


class AbstractNet9(AbstractNetwork):
    def __init__(self, net: NormalizedResnet, checker) -> None:
        self.normalize = AbstractNormalize(net.normalization)
        resnet: ResNet = net.resnet

        self.conv1 = AbstractConvolution(resnet[0])
        self.bn1 = AbstractBatchNorm(resnet[1])
        self.relu1 = AbstractReLU()

        blockLayers1: torch.nn.Sequential = resnet[3]
        self.block1 = AbstractBlockSubnet(blockLayers1[0], checker)
        self.relu2 = AbstractReLU()

        blockLayers2: torch.nn.Sequential = resnet[4]
        self.block2 = AbstractBlockSubnet(blockLayers2[0], checker)
        self.relu3 = AbstractReLU()

        self.flatten = AbstractFlatten()
        self.lin1 = AbstractLinear(resnet[6])
        self.relu4 = AbstractReLU()
        self.lin2 = AbstractLinear(resnet[8])
        self.final_atransformer = None  # built in forward

        self.checker = checker

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []
        self.checker.check_next(abstract_shape)

        abstract_shape = self.normalize.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.conv1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        # Do not append, backsub will be done from bn1
        # prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.bn1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.block1.forward(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.recheck(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.block2.forward(abstract_shape, prev_abstract_shapes)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu3.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.flatten.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin1.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu4.forward(abstract_shape).expand()
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)

        return abstract_shape

    def get_alphas(self) -> List[Tensor]:
        alphas = [
            self.relu1.real_alphas,
            self.block1.relu1b.real_alphas,
            self.relu2.real_alphas,
            self.block2.relu1b.real_alphas,
            self.relu3.real_alphas,
            self.relu4.real_alphas,
        ]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 6
        assert self.relu1.real_alphas.shape == updated_alphas[0].shape
        self.relu1.real_alphas = updated_alphas[0]
        assert self.block1.relu1b.real_alphas.shape == updated_alphas[1].shape
        self.block1.relu1b.real_alphas = updated_alphas[1]
        assert self.relu2.real_alphas.shape == updated_alphas[2].shape
        self.relu2.real_alphas = updated_alphas[2]
        assert self.block2.relu1b.real_alphas.shape == updated_alphas[3].shape
        self.block2.relu1b.real_alphas = updated_alphas[3]
        assert self.relu3.real_alphas.shape == updated_alphas[4].shape
        self.relu3.real_alphas = updated_alphas[4]
        assert self.relu4.real_alphas.shape == updated_alphas[5].shape
        self.relu4.real_alphas = updated_alphas[5]


class AbstractNet10(AbstractNetwork):
    def __init__(self, net: NormalizedResnet, checker) -> None:
        self.normalize = AbstractNormalize(net.normalization)
        resnet: ResNet = net.resnet

        self.conv1 = AbstractConvolution(resnet[0])
        self.relu1 = AbstractReLU()

        blockLayers1: torch.nn.Sequential = resnet[2]
        self.block1 = AbstractBlockSubnet(blockLayers1[0], checker)
        self.relu2 = AbstractReLU()
        self.block2 = AbstractBlockSubnet(blockLayers1[2], checker)
        self.relu3 = AbstractReLU()

        blockLayers2: torch.nn.Sequential = resnet[3]
        self.block3 = AbstractBlockSubnet(blockLayers2[0], checker)
        self.relu4 = AbstractReLU()
        self.block4 = AbstractBlockSubnet(blockLayers2[2], checker)
        self.relu5 = AbstractReLU()

        self.flatten = AbstractFlatten()
        self.lin1 = AbstractLinear(resnet[5])
        self.relu6 = AbstractReLU()
        self.lin2 = AbstractLinear(resnet[7])
        self.final_atransformer = None  # built in forward

        self.checker = checker

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []
        self.checker.check_next(abstract_shape)

        abstract_shape = self.normalize.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.conv1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.block1.forward(abstract_shape, prev_abstract_shapes)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.block2.forward(abstract_shape, prev_abstract_shapes)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu3.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.block3.forward(abstract_shape, prev_abstract_shapes)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu4.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.block4.forward(abstract_shape, prev_abstract_shapes)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu5.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.flatten.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin1.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu6.forward(abstract_shape).expand()
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)

        return abstract_shape

    def get_alphas(self) -> List[Tensor]:
        alphas = [
            self.relu1.real_alphas,
            self.block1.relu1b.real_alphas,
            self.relu2.real_alphas,
            self.block2.relu1b.real_alphas,
            self.relu3.real_alphas,
            self.block3.relu1b.real_alphas,
            self.relu4.real_alphas,
            self.block4.relu1b.real_alphas,
            self.relu5.real_alphas,
            self.relu6.real_alphas,
        ]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 10
        assert self.relu1.real_alphas.shape == updated_alphas[0].shape
        self.relu1.real_alphas = updated_alphas[0]
        assert self.block1.relu1b.real_alphas.shape == updated_alphas[1].shape
        self.block1.relu1b.real_alphas = updated_alphas[1]
        assert self.relu2.real_alphas.shape == updated_alphas[2].shape
        self.relu2.real_alphas = updated_alphas[2]
        assert self.block2.relu1b.real_alphas.shape == updated_alphas[3].shape
        self.block2.relu1b.real_alphas = updated_alphas[3]
        assert self.relu3.real_alphas.shape == updated_alphas[4].shape
        self.relu3.real_alphas = updated_alphas[4]
        assert self.block3.relu1b.real_alphas.shape == updated_alphas[5].shape
        self.block3.relu1b.real_alphas = updated_alphas[5]
        assert self.relu4.real_alphas.shape == updated_alphas[6].shape
        self.relu4.real_alphas = updated_alphas[6]
        assert self.block4.relu1b.real_alphas.shape == updated_alphas[7].shape
        self.block4.relu1b.real_alphas = updated_alphas[7]
        assert self.relu5.real_alphas.shape == updated_alphas[8].shape
        self.relu5.real_alphas = updated_alphas[8]
        assert self.relu6.real_alphas.shape == updated_alphas[9].shape
        self.relu6.real_alphas = updated_alphas[9]


class AbstractBlockSubnet(AbstractNetwork):
    def __init__(self, block: BasicBlock, checker) -> None:
        self.path_a: torch.nn.Sequential = block.path_a
        self.path_b: torch.nn.Sequential = block.path_b
        self.bn: bool = block.bn

        # Path B
        self.conv1b = AbstractConvolution(self.path_b[0])
        if self.bn:
            self.bn1b = AbstractBatchNorm(self.path_b[1])
            self.relu1b = AbstractReLU()
            self.conv2b = AbstractConvolution(self.path_b[3])
            self.bn2b = AbstractBatchNorm(self.path_b[4])
        else:
            self.bn1b = None
            self.relu1b = AbstractReLU()
            self.conv2b = AbstractConvolution(self.path_b[2])
            self.bn2b = None

        # Path A
        self.conv1a = self.create_abstract_id_conv(self.conv2b.c_out, self.conv1b.c_in)
        self.bn1a = None
        if len(self.path_a) > 1:
            # self.path_a[0] == Identity
            self.conv1a = AbstractConvolution(self.path_a[1])
            if self.bn:
                self.bn1a = AbstractBatchNorm(self.path_a[2])

        self.main_checker = checker
        self.checker_a = checker.next_path_a_checker()
        self.checker_b = checker.next_path_b_checker()
        

    def forward(
        self, abstract_shape: ReluAbstractShape, previous_shapes: List[AbstractShape]
    ) -> ResidualAbstractShape:
        self.checker_a.reset(self.main_checker.x)
        self.checker_b.reset(self.main_checker.x)
        prev_abstract_shapes_a = []
        prev_abstract_shapes_b = []
        abstract_shape_a = abstract_shape
        abstract_shape_b = abstract_shape
        # self.checker.check_next(abstract_shape)

        # Path A
        self.checker_a.check_next(abstract_shape_a)
        # conv1a
        abstract_shape_a = self.conv1a.forward(abstract_shape_a)
        self.checker_a.check_next(abstract_shape_a)
        # self.checker.check_next(abstract_shape_a)
        if self.bn:
            abstract_shape_a = self.bn1a.forward(abstract_shape_a)
            self.checker_a.check_next(abstract_shape_a)
            # self.checker.check_next(abstract_shape_a)
            prev_abstract_shapes_a.append(abstract_shape_a)
        else:
            prev_abstract_shapes_a.append(abstract_shape_a)

        # Path B
        self.checker_b.check_next(abstract_shape_b)
        # conv1b
        abstract_shape_b = self.conv1b.forward(abstract_shape_b)
        self.checker_b.check_next(abstract_shape_b)
        self.checker=self.checker_b
        abstract_shape_b = self.backsub(abstract_shape_b, previous_shapes, check=True)
        self.checker_b.recheck(abstract_shape_b)
        # self.checker.check_next(abstract_shape_b)
        if self.bn:
            abstract_shape_b = self.bn1b.forward(abstract_shape_b)
            self.checker_b.check_next(abstract_shape_b)
            # self.checker.check_next(abstract_shape_b)
            prev_abstract_shapes_b.append(abstract_shape_b)
        else:
            prev_abstract_shapes_b.append(abstract_shape_b)
        # relu1b
        abstract_shape_b = self.relu1b.forward(abstract_shape_b)
        self.checker_b.check_next(abstract_shape_b)
        # self.checker.check_next(abstract_shape_b)
        prev_abstract_shapes_b.append(abstract_shape_b)
        # conv2b
        abstract_shape_b = self.conv2b.forward(abstract_shape_b)
        self.checker_b.check_next(abstract_shape_b)
        # self.checker.check_next(abstract_shape_b)
        if self.bn:
            abstract_shape_b = self.bn2b.forward(abstract_shape_b)
            self.checker_b.check_next(abstract_shape_b)
            # self.checker.check_next(abstract_shape_b)
            prev_abstract_shapes_b.append(abstract_shape_b)
        else:
            prev_abstract_shapes_b.append(abstract_shape_b)

        # ResConnection
        lower = abstract_shape_a.lower + abstract_shape_b.lower
        upper = abstract_shape_a.upper + abstract_shape_b.upper
        abstract_shape = ResidualAbstractShape(
            torch.zeros(abstract_shape_b.y_greater.shape),
            torch.zeros(abstract_shape_b.y_greater.shape),
            lower, upper, *prev_abstract_shapes_a, *prev_abstract_shapes_b
        )
        # self.checker.check_next(abstract_shape)
        return abstract_shape

    def create_abstract_id_conv(self, c_out: int, c_in: int) -> AbstractConvolution:
        kernel = torch.zeros(c_out, c_in, 1, 1)
        kernel_diag = torch.diagonal(kernel, dim1=0, dim2=1)
        kernel_diag[...] = 1
        # kernel, bias, stride, padding
        return AbstractConvolution(kernel, None, 1, 0)
