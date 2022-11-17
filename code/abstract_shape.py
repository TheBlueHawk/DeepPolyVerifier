from tokenize import Double
from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

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
        return f"{self.y_greater}\n{self.y_less}\n{self.lower}\n{self.upper}"

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
        greater_backsub_cube = make_cube()
        bias_greater = self.y_greater[:, 0]
        weights_greater = self.y_greater[:, 1:]
        new_greater_with_extra = torch.tensordot(weights_greater, 
                                        greater_backsub_cube, dims=([1],[1]))
        new_greater = torch.diagonal(new_greater_with_extra, dim1=0, dim2=1).T

        # Add existing bias to new bias
        new_greater[:, 0] += bias_greater

        less_backsub_cube = make_cube()
        bias_less = self.y_less[:, 0]
        weights_less = self.y_less[:, 1:]
        new_less_with_extra = torch.tensordot(weights_less, 
                                            less_backsub_cube, dims=([1],[1]))
        # new_less = torch.stack([new_less_with_extra[i, i] 
        #                         for i in range(weights_less.shape[0])],
        #                         axis=0)
        new_less = torch.diagonal(new_less_with_extra, dim1=0, dim2=1).T
        # Add existing bias to new bias
        new_less[:, 0] += bias_less

        # TODO Not sure which abstract shape we will need here
        # To update lower, upper, do a forward pass with the new weights
        return LinearAbstractShape(new_greater, new_less, None, None)


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

        return AbstractShape(
            y_greater,
            y_less,
            self.lower.clone(),
            self.upper.clone())

    def backsub(self, previous_abstract_shape):
        expanded_self = self.expand()
        return expanded_self.backsub(previous_abstract_shape)


def create_abstract_input_shape(inputs, eps):
    return AbstractShape(
        y_greater=torch.clamp(inputs-eps, 0, 1),
        y_less=torch.clamp(inputs+eps, 0, 1),
        lower=torch.clamp(inputs-eps, 0, 1),
        upper=torch.clamp(inputs+eps, 0, 1)
    )

def make_cube():
    """Assuming that the cube is of shape <N, N-1, N-2 + 1>"""
    return torch.ones(2, 3, 5)

def main():
    cur_shape = LinearAbstractShape(
        torch.ones(2, 4),
        torch.ones(2, 4),
        torch.ones(2),
        torch.ones(2),
    )
    prev_shape = ReluAbstractShape(
        torch.ones(3, 5),
        torch.ones(3, 5),
        torch.ones(3),
        torch.ones(3),
    )

    print(cur_shape.backsub(prev_shape))

if __name__ == '__main__':
    main()