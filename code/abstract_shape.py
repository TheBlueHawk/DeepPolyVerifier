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
        greater_backsub_cube = buildConstraints3DMatrix(self, previous_abstract_shape)
        bias_greater = self.y_greater[:, 0]
        weights_greater = self.y_greater[:, 1:].unsqueeze(1)
        new_greater = (weights_greater @ greater_backsub_cube).squeeze()
        # Add existing bias to new bias
        new_greater[:, 0] += bias_greater

        less_backsub_cube = buildConstraints3DMatrix(self, previous_abstract_shape)
        bias_less = self.y_less[:, 0]
        weights_less = self.y_less[:, 1:].unsqueeze(1)
        new_less = (weights_less @ less_backsub_cube).squeeze()
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

        return AbstractShape(y_greater, y_less, self.lower.clone(), self.upper.clone())

    def backsub(self, previous_abstract_shape):
        expanded_self = self.expand()
        return expanded_self.backsub(previous_abstract_shape)


def create_abstract_input_shape(inputs, eps):
    return AbstractShape(
        y_greater=torch.clamp(inputs - eps, 0, 1),
        y_less=torch.clamp(inputs + eps, 0, 1),
        lower=torch.clamp(inputs - eps, 0, 1),
        upper=torch.clamp(inputs + eps, 0, 1),
    )


def buildConstraints3DMatrix(
    current_layer_ashape: AbstractShape, previous_layer_ashape: AbstractShape
) -> Tensor:
    curr_y_greater = current_layer_ashape.y_greater  # shape: [N, N1+1]
    prev_y_greater = previous_layer_ashape.y_greater  # shape: [N1, N2+1]
    prev_y_smaller = previous_layer_ashape.y_less  # shape: [N1, N2+1]
    # N = curr_y_greater.shape[0]  # n_neurons_current_layer
    # N1 = curr_y_greater.shape[1]  # n_neurons_prev_layer
    # N2 = prev_y_greater.shape[1]  # n_neurons_prev_prev_layer
    assert (
        curr_y_greater.shape[1] - 1
        == prev_y_smaller.shape[0]
        == prev_y_greater.shape[0]
    )
    # curr_b = curr_y_greater[:, 0].unsqueeze(1)  # shape: [N, 1, 1]
    # bias = torch.concat((curr_b, torch.zeros(N, N2, 1)), dim=1)  # shape: [N, N2+1, 1]
    alphas = curr_y_greater[:, 1:].unsqueeze(1)  # shape: [N, 1, N1]
    cube = torch.where(
        alphas.swapdims(1, 2) >= 0, prev_y_greater, prev_y_smaller
    )  # shape: [N, N1, N2 + 1]
    return cube


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
