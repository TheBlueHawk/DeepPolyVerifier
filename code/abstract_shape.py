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
    layer and the previous layer + the intercept.

    Attributes:
        y_greater: tensor of shape <no. curr neurons, no. prev neurons + 1>
        y_less: tensor of shape <no. curr neurons,  no. prev neurons + 1>
        lower: tensor of shape <no. curr neurons>
        upper: tensor if shape <no. curr neurons>
    """

    def backsub(self, previous_abstract_shape):
        pass


class ReluAbstractShape(AbstractShape):
    """The compact representation of the output of the abstract ReLU layer.

    y_greater stores only the slope of the line, as the intercept is always 0.
    y_less stores only the slope and intercept of the line.

    Attributes:
        y_greater: tensor of shape <no. curr neurons, 1>
        y_less: tensor of shape <no. curr neurons, 2>
        lower: tensor of shape <no. curr neurons>
        upper: tensor if shape <<no. curr neurons>
    """

    def expand(self):
        """Rerepresents a compact relu abstract shape wrt. all neurons in the prevoious layer"""
        
        intercept_greater = torch.zeros_like(self.y_greater)
        weights_greater = self.y_greater.flatten()
        weights_greater_expanded = torch.diag(weights_greater)
        y_greater = torch.cat([intercept_greater, weights_greater_expanded], axis=1)

        intercept_less = self.y_less[:, 0:1]
        weights_less = self.y_less[:, 1]
        weights_less_expanded = torch.diag(weights_less)
        y_less = torch.cat([intercept_less, weights_less_expanded], axis=1)

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

def main():
    pass

if __name__ == '__main__':
    main()