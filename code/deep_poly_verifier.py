import torch
from torch import Tensor

from abstract_shape import create_abstract_input_shape, AbstractShape
from abstract_networks import AbstractNet1
from transformers import AbstractLinear


def get_anet_class_from_name(net_name):
    abstract_nets = {"net1": AbstractNet1}
    return abstract_nets[net_name]


class DeepPolyVerifier:
    def __init__(self, net, net_name, use_final_layer=False):
        abstract_net_class = get_anet_class_from_name(net_name)
        self.abstract_net = abstract_net_class(net)
        self.N = 10

    def verify(self, inputs, eps, true_label):
        abstract_input = create_abstract_input_shape(inputs, eps)

        curr_abstract_shape = abstract_input
        for abstract_transformer in self.abstract_net.get_abstract_transformers():
            curr_abstract_shape = abstract_transformer.forward(curr_abstract_shape)
        return finalLayerVerification(curr_abstract_shape, true_label, self.N)


def addFinalLayerWeights(true_lablel: int, N: int) -> Tensor:
    weights = torch.zeros(N, N)
    for i in range(N):
        if i == true_lablel:
            weights.T[i] = torch.ones(N)
        weights[i][i] += -1

    bias = torch.zeros(N, 1)
    wb = torch.cat((bias, weights), dim=-1)
    return wb


def verifyFinalShape(final_shape: AbstractShape) -> bool:
    l = final_shape.lower
    return torch.all(torch.greater_equal(l, torch.zeros_like(l))).item()


def finalLayerVerification(
    current_abstract_shape: AbstractShape, true_label: int, N: int
) -> bool:
    final_layer: AbstractLinear = AbstractLinear(addFinalLayerWeights(true_label, N))
    final_shape = final_layer.forward(current_abstract_shape)
    v = verifyFinalShape(final_shape)
    return v


def buildConstraints3DMatrix(
    current_layer_ashape: AbstractShape, previous_layer_ashape: AbstractShape
) -> Tensor:
    curr_y_greater = current_layer_ashape.y_greater  # shape: [N, N1+1]
    prev_y_greater = previous_layer_ashape.y_greater  # shape: [N1, N2+1]
    prev_y_smaller = previous_layer_ashape.y_less  # shape: [N1, N2+1]
    N = curr_y_greater.shape[0]  # n_neurons_current_layer
    N1 = curr_y_greater.shape[1]  # n_neurons_prev_layer
    N2 = prev_y_greater.shape[1]  # n_neurons_prev_prev_layer
    assert curr_y_greater.shape[1] == prev_y_smaller.shape[0] == prev_y_greater.shape[0]
    curr_b = curr_y_greater[:, :, 0].unsqueeze(1)  # shape: [N, 1, 1]
    bias = torch.concat((curr_b, torch.zeros(N, N2, 1)), dim=1)  # shape: [N, N2+1, 1]
    alphas = curr_y_greater[:, :, 1:].unsqueeze(1)  # shape: [N, 1, N1]
    cube = torch.where(
        alphas.swapdims(1, 2) >= 0, prev_y_greater, prev_y_smaller
    )  # shape: [N, N1, N2 + 1]
    return cube


def main():
    aInput = AbstractShape(
        Tensor([[1, 1], [0, 1]]),
        Tensor([[0, 1], [0, 1]]),
        Tensor([4, -2]),
        Tensor([6, 2]),
    )
    b = finalLayerVerification(aInput, 0, 2)
    assert b == True


if __name__ == "__main__":
    main()
