from abstract_shape import AbstractShape
from typing import List
from torch import Tensor
import torch
from transformers import (
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

    def backsub(self, abstract_shape, previous_shapes_list, first_abstract_shape):
        curr_ashape = abstract_shape
        for previous_shape in reversed(previous_shapes_list):
            curr_ashape = curr_ashape.backsub(previous_shape)

        # Recompute l & u
        tmp_ashape_u = AbstractLinear(curr_ashape.y_greater).forward(first_abstract_shape)
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
        final_layer: AbstractLinear = AbstractLinear(self.buildFinalLayerWeights(true_label, N))
        return final_layer


class AbstractNet1(AbstractNetwork):
    def __init__(self, net) -> None:
        self.flatten = AbstractFlatten()
        self.normalize = AbstractNormalize(net.layers[0])
        self.lin1 = AbstractLinear(net.layers[2])
        self.relu1 = AbstractReLU()
        self.lin2 = AbstractLinear(net.layers[4])
        self.final_atransformer = None # built in forward

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []

        # No need to backsub Normalization and Flatten, they operate pointwise
        abstract_shape = self.flatten.forward(abstract_shape)
        abstract_shape = self.normalize.forward(abstract_shape)
        first_abstract_shape = abstract_shape

        # Won't get tighter l, u after backsubstituting the first linear layer
        abstract_shape = self.lin1.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape).expand()
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes, 
                                first_abstract_shape)

        return abstract_shape


class AbstractNet2(AbstractNetwork):
    def __init__(self, net) -> None:
        self.flatten = AbstractFlatten()
        self.normalize = AbstractNormalize(net.layers[0])
        self.lin1 = AbstractLinear(net.layers[2])
        self.relu1 = AbstractReLU()
        self.lin2 = AbstractLinear(net.layers[4])
        self.relu2 = AbstractReLU()
        self.lin3 = AbstractLinear(net.layers[6])
        self.final_atransformer = None # built in forward

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []

        # No need to backsub Normalization and Flatten, they operate pointwise
        abstract_shape = self.flatten.forward(abstract_shape)
        abstract_shape = self.normalize.forward(abstract_shape)
        first_abstract_shape = abstract_shape

        # Won't get tighter l, u after backsubstituting the first linear layer
        abstract_shape = self.lin1.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape).expand()
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes,
                                first_abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape).expand()
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin3.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes,
                                first_abstract_shape)

        return abstract_shape


class AbstractNet3(AbstractNetwork):
    def __init__(self, net) -> None:
        self.flatten = AbstractFlatten()
        self.normalize = AbstractNormalize(net.layers[0])
        self.lin1 = AbstractLinear(net.layers[2])
        self.relu1 = AbstractReLU()
        self.lin2 = AbstractLinear(net.layers[4])
        self.relu2 = AbstractReLU()
        self.lin3 = AbstractLinear(net.layers[6])
        self.final_atransformer = None # built in forward

    def forward(self, abstract_shape, true_label, N):
        self.final_atransformer = self.buildFinalLayer(true_label, N)
        prev_abstract_shapes = []

        # No need to backsub Normalization and Flatten, they operate pointwise
        abstract_shape = self.flatten.forward(abstract_shape)
        abstract_shape = self.normalize.forward(abstract_shape)
        first_abstract_shape = abstract_shape

        # Won't get tighter l, u after backsubstituting the first linear layer
        abstract_shape = self.lin1.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu1.forward(abstract_shape).expand()
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin2.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes,
                                first_abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.relu2.forward(abstract_shape).expand()
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.lin3.forward(abstract_shape)
        prev_abstract_shapes.append(abstract_shape)

        abstract_shape = self.final_atransformer.forward(abstract_shape)
        abstract_shape = self.backsub(abstract_shape, prev_abstract_shapes,
                                first_abstract_shape)

        return abstract_shape