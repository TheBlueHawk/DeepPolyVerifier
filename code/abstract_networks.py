from abstract_shape import AbstractShape, ConvAbstractShape, LinearAbstractShape
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
    AbstractResidualSum,
)


class AbstractNetwork:
    def __init__(
        self,
        abstract_transformers: List,
    ) -> None:
        self.abstract_transformers = abstract_transformers

    def recompute_bounds_linear(self, first_ashape, curr_ashape):
        tmp_ashape_u = AbstractLinear(curr_ashape.y_greater).forward(first_ashape)
        tmp_ashape_l = AbstractLinear(curr_ashape.y_less).forward(first_ashape)
        return tmp_ashape_u.upper, tmp_ashape_l.lower

    def recompute_bounds_conv(self, first_ashape, curr_ashape):
        # _init_from_tensor(self, kernel, bias, stride, padding, dilation=(1, 1)):
        kernel = curr_ashape.y_greater[
            :, :, :, 1:
        ]  # <C, N, N, C2 * composedK * composedK>
        kernel = kernel.flatten(0, 2)  # <C * N * N, C2 * composedK * composedK>
        kernel = kernel.reshape(
            kernel.shape[0], curr_ashape.c_in, curr_ashape.k, curr_ashape.k
        )  # <C * N * N, C2, composedK, composedK> = [c_out, c_in, k,k]
        bias = curr_ashape.y_greater[:, :, :, 0].flatten()  # <C * N * N>
        temp_ashape = AbstractConvolution(
            kernel,
            bias,
            curr_ashape.stride,
            curr_ashape.padding,
        ).forward(first_ashape)

        new_u = temp_ashape.upper  # <C * N * N, N, N>  # overcomputation ...
        new_l = temp_ashape.lower  # <C * N * N, N, N> # ... need to select idx
        new_N = new_u.shape[-1]
        new_u = (
            new_u.reshape(-1, new_N, new_N, new_N, new_N)
            .diagonal(dim1=1, dim2=3)
            .diagonal(dim1=1, dim2=2)
        )
        new_l = (
            new_l.reshape(-1, new_N, new_N, new_N, new_N)
            .diagonal(dim1=1, dim2=3)
            .diagonal(dim1=1, dim2=2)
        )

        return new_u, new_l

    def backsub(self, abstract_shape: AbstractShape, previous_shapes) -> AbstractShape:
        curr_ashape = abstract_shape
        for previous_shape in reversed(previous_shapes[1:]):
            # TODO: expand prev shape size (N -> N + 2) with padding
            if isinstance(curr_ashape, ConvAbstractShape):
                curr_ashape.zero_out_padding_weights()
            curr_ashape = curr_ashape.backsub(previous_shape)

        # Recompute l & u
        if isinstance(curr_ashape, LinearAbstractShape):
            abstract_shape.upper, abstract_shape.lower = self.recompute_bounds_linear(
                previous_shapes[0], curr_ashape
            )

        elif isinstance(curr_ashape, ConvAbstractShape):
            abstract_shape.upper, abstract_shape.lower = self.recompute_bounds_conv(
                previous_shapes[0], curr_ashape
            )
            if isinstance(abstract_shape, LinearAbstractShape):
                abstract_shape.upper = torch.squeeze(abstract_shape.upper)
                abstract_shape.lower = torch.squeeze(abstract_shape.lower)

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
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)

        return abstract_shape

    def get_alphas(self) -> List[Tensor]:
        alphas = [self.relu1.alphas]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 1
        assert self.relu1.alphas.shape == updated_alphas[0].shape
        self.relu1.alphas = updated_alphas[0]


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
        alphas = [self.relu1.alphas, self.relu2.alphas]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 2
        assert self.relu1.alphas.shape == updated_alphas[0].shape
        self.relu1.alphas = updated_alphas[0]
        assert self.relu2.alphas.shape == updated_alphas[1].shape
        self.relu2.alphas = updated_alphas[1]


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
        alphas = [self.relu1.alphas, self.relu2.alphas]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 2
        assert self.relu1.alphas.shape == updated_alphas[0].shape
        self.relu1.alphas = updated_alphas[0]
        assert self.relu2.alphas.shape == updated_alphas[1].shape
        self.relu2.alphas = updated_alphas[1]


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
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
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
        alphas = [self.relu1.alphas, self.relu2.alphas]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 2
        assert self.relu1.alphas.shape == updated_alphas[0].shape
        self.relu1.alphas = updated_alphas[0]
        assert self.relu2.alphas.shape == updated_alphas[1].shape
        self.relu2.alphas = updated_alphas[1]


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
        alphas = [self.relu1.alphas, self.relu2.alphas, self.relu3.alphas]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 3
        assert self.relu1.alphas.shape == updated_alphas[0].shape
        self.relu1.alphas = updated_alphas[0]
        assert self.relu2.alphas.shape == updated_alphas[1].shape
        self.relu2.alphas = updated_alphas[1]
        assert self.relu3.alphas.shape == updated_alphas[2].shape
        self.relu3.alphas = updated_alphas[2]


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
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)

        return abstract_shape

    def get_alphas(self) -> List[Tensor]:
        alphas = [self.relu1.alphas, self.relu2.alphas, self.relu3.alphas]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 3
        assert self.relu1.alphas.shape == updated_alphas[0].shape
        self.relu1.alphas = updated_alphas[0]
        assert self.relu2.alphas.shape == updated_alphas[1].shape
        self.relu2.alphas = updated_alphas[1]
        assert self.relu3.alphas.shape == updated_alphas[2].shape
        self.relu3.alphas = updated_alphas[2]


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
            self.relu1.alphas,
            self.relu2.alphas,
            self.relu3.alphas,
            self.relu4.alphas,
        ]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 4
        assert self.relu1.alphas.shape == updated_alphas[0].shape
        self.relu1.alphas = updated_alphas[0]
        assert self.relu2.alphas.shape == updated_alphas[1].shape
        self.relu2.alphas = updated_alphas[1]
        assert self.relu3.alphas.shape == updated_alphas[2].shape
        self.relu3.alphas = updated_alphas[2]
        assert self.relu4.alphas.shape == updated_alphas[3].shape
        self.relu4.alphas = updated_alphas[3]


class AbstractNet8(AbstractNetwork):
    def __init__(self, net, checker) -> None:
        self.conv1 = AbstractConvolution(net.layers[1])
        self.relu1 = AbstractReLU()

        blockLayers1: torch.nn.Sequential = net.layers[3][0]
        self.block1 = AbstractBlockSubnet(blockLayers1.layer[0])
        self.relu2 = AbstractReLU()
        self.block2 = AbstractBlockSubnet(blockLayers1.layer[3])
        self.relu3 = AbstractReLU()

        self.flatten = AbstractFlatten()
        self.lin1 = AbstractLinear(net.layers[5])
        self.relu4 = AbstractReLU()
        self.lin2 = AbstractLinear(net.layers[7])
        self.final_atransformer = None  # built in forward

        self.checker = checker

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []
        self.checker.check_next(abstract_shape)

        abstract_shape = self.conv1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.block1.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.block2.forward(abstract_shape)
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
            self.relu1.alphas,
            self.block1.relu1b.alphas,
            self.relu2.alphas,
            self.block2.relu1b.alphas,
            self.relu3.alphas,
            self.relu4.alphas,
        ]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 6
        assert self.relu1.alphas.shape == updated_alphas[0].shape
        self.relu1.alphas = updated_alphas[0]
        assert self.block1.relu1b.alphas.shape == updated_alphas[1].shape
        self.block1.relu1b.alphas.shape = updated_alphas[1]
        assert self.relu2.alphas.shape == updated_alphas[2].shape
        self.relu2.alphas = updated_alphas[2]
        assert self.block2.relu1b.alphas.shape == updated_alphas[3].shape
        self.block2.relu1b.alphas.shape = updated_alphas[3]
        assert self.relu3.alphas.shape == updated_alphas[4].shape
        self.relu3.alphas = updated_alphas[4]
        assert self.relu4.alphas.shape == updated_alphas[5].shape
        self.relu4.alphas = updated_alphas[5]


class AbstractNet9(AbstractNetwork):
    def __init__(self, net, checker) -> None:
        self.conv1 = AbstractConvolution(net.layers[1])
        self.bn1 = AbstractBatchNorm(net.layers[2])
        self.relu1 = AbstractReLU()

        blockLayers1: torch.nn.Sequential = net.layers[4][0]
        self.block1 = AbstractBlockSubnet(blockLayers1.layer[0])
        self.relu2 = AbstractReLU()

        blockLayers2: torch.nn.Sequential = net.layers[4][0]
        self.block2 = AbstractBlockSubnet(blockLayers2.layer[0])
        self.relu3 = AbstractReLU()

        self.flatten = AbstractFlatten()
        self.lin1 = AbstractLinear(net.layers[6])
        self.relu4 = AbstractReLU()
        self.lin2 = AbstractLinear(net.layers[8])
        self.final_atransformer = None  # built in forward

        self.checker = checker

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []
        self.checker.check_next(abstract_shape)

        abstract_shape = self.conv1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.bn1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.block1.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.block2.forward(abstract_shape)
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
            self.relu1.alphas,
            self.block1.relu1b.alphas,
            self.relu2.alphas,
            self.block2.relu1b.alphas,
            self.relu3.alphas,
            self.relu4.alphas,
        ]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 6
        assert self.relu1.alphas.shape == updated_alphas[0].shape
        self.relu1.alphas = updated_alphas[0]
        assert self.block1.relu1b.alphas.shape == updated_alphas[1].shape
        self.block1.relu1b.alphas.shape = updated_alphas[1]
        assert self.relu2.alphas.shape == updated_alphas[2].shape
        self.relu2.alphas = updated_alphas[2]
        assert self.block2.relu1b.alphas.shape == updated_alphas[3].shape
        self.block2.relu1b.alphas.shape = updated_alphas[3]
        assert self.relu3.alphas.shape == updated_alphas[4].shape
        self.relu3.alphas = updated_alphas[4]
        assert self.relu4.alphas.shape == updated_alphas[5].shape
        self.relu4.alphas = updated_alphas[5]


class AbstractNet10(AbstractNetwork):
    def __init__(self, net, checker) -> None:
        self.conv1 = AbstractConvolution(net.layers[1])
        self.relu1 = AbstractReLU()

        blockLayers1: torch.nn.Sequential = net.layers[3][0]
        self.block1 = AbstractBlockSubnet(blockLayers1.layer[0])
        self.relu2 = AbstractReLU()
        self.block2 = AbstractBlockSubnet(blockLayers1.layer[3])
        self.relu3 = AbstractReLU()

        blockLayers2: torch.nn.Sequential = net.layers[4][0]
        self.block3 = AbstractBlockSubnet(blockLayers2.layer[0])
        self.relu4 = AbstractReLU()
        self.block4 = AbstractBlockSubnet(blockLayers2.layer[3])
        self.relu5 = AbstractReLU()

        self.flatten = AbstractFlatten()
        self.lin1 = AbstractLinear(net.layers[6])
        self.relu6 = AbstractReLU()
        self.lin2 = AbstractLinear(net.layers[8])
        self.final_atransformer = None  # built in forward

        self.checker = checker

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []
        self.checker.check_next(abstract_shape)

        abstract_shape = self.conv1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.block1.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.block2.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu3.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.block3.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu4.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.block4.forward(abstract_shape)
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
            self.relu1.alphas,
            self.block1.relu1b.alphas,
            self.relu2.alphas,
            self.block2.relu1b.alphas,
            self.relu3.alphas,
            self.block3.relu1b.alphas,
            self.relu4.alphas,
            self.block4.relu1b.alphas,
            self.relu5.alphas,
            self.relu6.alphas,
        ]
        return alphas

    def set_alphas(self, updated_alphas: List[Tensor]) -> List[Tensor]:
        assert len(updated_alphas) == 10
        assert self.relu1.alphas.shape == updated_alphas[0].shape
        self.relu1.alphas = updated_alphas[0]
        assert self.block1.relu1b.alphas.shape == updated_alphas[1].shape
        self.block1.relu1b.alphas.shape = updated_alphas[1]
        assert self.relu2.alphas.shape == updated_alphas[2].shape
        self.relu2.alphas = updated_alphas[2]
        assert self.block2.relu1b.alphas.shape == updated_alphas[3].shape
        self.block2.relu1b.alphas.shape = updated_alphas[3]
        assert self.relu3.alphas.shape == updated_alphas[4].shape
        self.relu3.alphas = updated_alphas[4]
        assert self.block3.relu1b.alphas.shape == updated_alphas[5].shape
        self.block3.relu1b.alphas.shape = updated_alphas[5]
        assert self.relu4.alphas.shape == updated_alphas[6].shape
        self.relu4.alphas = updated_alphas[6]
        assert self.block4.relu1b.alphas.shape == updated_alphas[7].shape
        self.block4.relu1b.alphas.shape = updated_alphas[7]
        assert self.relu5.alphas.shape == updated_alphas[8].shape
        self.relu5.alphas = updated_alphas[8]
        assert self.relu6.alphas.shape == updated_alphas[9].shape
        self.relu6.alphas = updated_alphas[9]


class AbstractBlockSubnet(AbstractNetwork):
    def __init__(self, block: BasicBlock, checker) -> None:
        self.path_a: torch.nn.Sequential = block.path_a
        self.path_b: torch.nn.Sequential = block.path_b
        self.bn: bool = block.bn

        # Path A
        self.conv1a = None
        self.bn1a = None
        if len(self.path_a) > 1:
            # self.path_a[1] == Identity
            self.conv1a = AbstractConvolution(self.path_a[1])
            if self.bn:
                self.bn1a = AbstractBatchNorm(self.path_a[2])

        # Path B
        self.conv1b = AbstractConvolution(self.path_b[0])
        if self.bn:
            self.bn1b = AbstractBatchNorm(self.path_b[1])
            self.relu1b = AbstractReLU(self.path_b[2])
            self.conv2b = AbstractConvolution(self.path_b[3])
            self.bn2b = AbstractBatchNorm(self.path_b[4])
        else:
            self.relu1b = AbstractReLU(self.path_b[1])
            self.conv1b = AbstractConvolution(self.path_b[2])

        self.res_sum = AbstractResidualSum()
        self.checker = checker

    def forward(self, abstract_shape: AbstractShape) -> AbstractShape:
        prev_abstract_shapes_a = []
        prev_abstract_shapes_b = []
        abstract_shape_a = abstract_shape
        abstract_shape_b = abstract_shape
        self.checker.check_next(abstract_shape)

        # Path A
        if self.conv1a is not None:
            # conv1a
            abstract_shape_a = self.conv1a.forward(abstract_shape_a)
            self.checker.check_next(abstract_shape_a)
            prev_abstract_shapes_a.append(abstract_shape_a)
            # bn1a
            if self.bn:
                abstract_shape_a = self.bn1a.forward(abstract_shape_a)
                self.checker.check_next(abstract_shape_a)
                prev_abstract_shapes_a.append(abstract_shape_a)

        # Path B
        # conv1b
        abstract_shape_b = self.conv1b.forward(abstract_shape_b)
        self.checker.check_next(abstract_shape_b)
        prev_abstract_shapes_b.append(abstract_shape_b)
        # bn1b
        if self.bn:
            abstract_shape_b = self.bn1b.forward(abstract_shape_b)
            self.checker.check_next(abstract_shape_b)
            prev_abstract_shapes_b.append(abstract_shape_b)
        # relu1b
        abstract_shape_b = self.relu1b.forward(abstract_shape_b)
        self.checker.check_next(abstract_shape_b)
        prev_abstract_shapes_b.append(abstract_shape_b)
        # conv2b
        abstract_shape_b = self.conv2b.forward(abstract_shape_b)
        abstract_shape_b = self.backsub(abstract_shape_b, prev_abstract_shapes_b)
        self.checker.check_next(abstract_shape_b)
        prev_abstract_shapes_b.append(abstract_shape_b)
        if self.bn:
            abstract_shape_b = self.bn2b.forward(abstract_shape_b)
            self.checker.check_next(abstract_shape_b)
            prev_abstract_shapes_b.append(abstract_shape_b)

        # ResConnection
        abstract_shape = self.res_sum.forward(abstract_shape_a, abstract_shape_b)
        self.checker.check_next(abstract_shape)
        return abstract_shape
