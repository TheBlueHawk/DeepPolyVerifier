from abstract_shape import AbstractShape
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

    def backsub(
        self, abstract_shape, previous_shapes_list, first_abstract_shape
    ) -> AbstractShape:
        curr_ashape = abstract_shape
        for previous_shape in reversed(previous_shapes_list):
            curr_ashape = curr_ashape.backsub(previous_shape)

        # Recompute l & u
        tmp_ashape_u = AbstractLinear(curr_ashape.y_greater).forward(
            first_abstract_shape
        )
        abstract_shape.upper = tmp_ashape_u.upper
        tmp_ashape_l = AbstractLinear(curr_ashape.y_less).forward(first_abstract_shape)
        abstract_shape.lower = tmp_ashape_l.lower

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
        first_abstract_shape = abstract_shape

        # Won't get tighter l, u after backsubstituting the first linear layer
        abstract_shape = self.lin1.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape).expand()
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(
            abstract_shape, prev_abstract_shapes, first_abstract_shape
        )

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
        first_abstract_shape = abstract_shape

        # Won't get tighter l, u after backsubstituting the first linear layer
        abstract_shape = self.lin1.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape).expand()
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        abstract_shape = self.backsub(
            abstract_shape, prev_abstract_shapes, first_abstract_shape
        )
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape).expand()
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin3.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(
            abstract_shape, prev_abstract_shapes, first_abstract_shape
        )

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
        first_abstract_shape = abstract_shape

        # Won't get tighter l, u after backsubstituting the first linear layer
        abstract_shape = self.lin1.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape).expand()
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        abstract_shape = self.backsub(
            abstract_shape, prev_abstract_shapes, first_abstract_shape
        )
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape).expand()
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin3.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(
            abstract_shape, prev_abstract_shapes, first_abstract_shape
        )

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

        assert self.checker.check_next(abstract_shape)

        # No need to backsub Normalization, it operates pointwise
        abstract_shape = self.normalize.forward(abstract_shape)
        assert self.checker.check_next(abstract_shape)
        first_abstract_shape = abstract_shape

        # Won't get tighter l, u after backsubstituting the first conv layer
        abstract_shape = self.conv1.forward(abstract_shape)
        assert self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape)
        assert self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.flatten.forward(abstract_shape)
        assert self.checker.check_next(abstract_shape)

        abstract_shape = self.lin1.forward(abstract_shape)
        assert self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape).expand()
        assert self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        assert self.checker.check_next(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        # abstract_shape = self.backsub(
        #     abstract_shape, prev_abstract_shapes, first_abstract_shape
        # )

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


class ANetChecker():
    def __init__(self, net):
        self.layers = net.layers
        self.current_layer = -1

    def reset(self, x):
        self.x = x

    def check_next(self, abstract_shape):
        if self.current_layer > -1:
            self.x = self.layers[self.current_layer](self.x)

        self.current_layer += 1
        return self.x_in_ashape(self.x, abstract_shape)

    def x_in_ashape(self, x, abstract_shape):
        raise NotImplementedError()


class DummyANetChecker(ANetChecker):
    def __init__(self, net):
        pass

    def reset(self, x):
        pass

    def check_next(self, abstract_shape):
        return True

class InclusionANetChecker(ANetChecker):
    def x_in_ashape(self, x, abstract_shape):
        return (torch.all(abstract_shape.lower <= x) and 
                torch.all(x <= abstract_shape.upper))

class AbstractNet5(AbstractNetwork):
    def __init__(self, net) -> None:
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

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []

        # No need to backsub Normalization, it operates pointwise
        abstract_shape = self.normalize.forward(abstract_shape)
        first_abstract_shape = abstract_shape

        # Won't get tighter l, u after backsubstituting the first linear layer
        abstract_shape = self.conv1.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.conv2.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.flatten.forward(abstract_shape)

        abstract_shape = self.lin1.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu3.forward(abstract_shape).expand()
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        # abstract_shape = self.backsub(
        #     abstract_shape, prev_abstract_shapes, first_abstract_shape
        # )

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
    def __init__(self, net) -> None:
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

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []

        # No need to backsub Normalization, it operates pointwise
        abstract_shape = self.normalize.forward(abstract_shape)
        first_abstract_shape = abstract_shape

        # Won't get tighter l, u after backsubstituting the first linear layer
        abstract_shape = self.conv1.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.conv2.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.flatten.forward(abstract_shape)

        abstract_shape = self.lin1.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu3.forward(abstract_shape).expand()
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        # abstract_shape = self.backsub(
        #     abstract_shape, prev_abstract_shapes, first_abstract_shape
        # )

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
    def __init__(self, net) -> None:
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

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []

        # No need to backsub Normalization, it operates pointwise
        abstract_shape = self.normalize.forward(abstract_shape)
        first_abstract_shape = abstract_shape

        # Won't get tighter l, u after backsubstituting the first linear layer
        abstract_shape = self.conv1.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.conv2.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.flatten.forward(abstract_shape)

        abstract_shape = self.lin1.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu3.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu4.forward(abstract_shape).expand()
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin3.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        # abstract_shape = self.backsub(
        #     abstract_shape, prev_abstract_shapes, first_abstract_shape
        # )

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
