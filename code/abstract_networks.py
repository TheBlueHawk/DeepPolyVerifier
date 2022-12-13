from abstract_shape import AbstractShape, ConvAbstractShape, LinearAbstractShape
from typing import List
from torch import Tensor
import torch
from transformers import (
    AbstractConvolution,
    AbstractNormalize,
    AbstractFlatten,
    AbstractLinear,
    AbstractReLU,
)


class AbstractNetwork:
    def __init__(
        self,
        abstract_transformers: List,
    ) -> None:
        self.abstract_transformers = abstract_transformers

    def backsub(self, abstract_shape: AbstractShape, previous_shapes) -> AbstractShape:
        curr_ashape = abstract_shape
        for previous_shape in reversed(previous_shapes[1:]):
            # TODO: expand prev shape size (N -> N + 2) with padding
            curr_ashape = curr_ashape.backsub(previous_shape)

        # Recompute l & u
        if isinstance(curr_ashape, LinearAbstractShape):
            tmp_ashape_u = AbstractLinear(curr_ashape.y_greater).forward(
                previous_shapes[0]
            )
            tmp_ashape_l = AbstractLinear(curr_ashape.y_less).forward(
                previous_shapes[0]
            )
            abstract_shape.upper = tmp_ashape_u.upper
            abstract_shape.lower = tmp_ashape_l.lower

        elif isinstance(curr_ashape, ConvAbstractShape):
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
            ).forward(previous_shapes[0])

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

            if isinstance(abstract_shape, LinearAbstractShape):
                new_u = torch.squeeze(new_u)
                new_l = torch.squeeze(new_l)

            abstract_shape.upper = new_u
            abstract_shape.lower = new_l
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
        abstract_shape = self.backsub(
            abstract_shape, prev_abstract_shapes
        )

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
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu3.forward(abstract_shape).expand()
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)


        abstract_shape = self.relu4.forward(abstract_shape).expand()
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin3.forward(abstract_shape)
        self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(
            abstract_shape, prev_abstract_shapes
        )

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
