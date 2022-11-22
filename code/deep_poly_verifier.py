import torch
from torch import Tensor

from abstract_shape import create_abstract_input_shape, AbstractShape, weightedLoss
from abstract_networks import AbstractNet1, AbstractNet2, AbstractNet3, AbstractNetwork
from transformers import AbstractLinear


def get_anet_class_from_name(net_name) -> AbstractNetwork:
    abstract_nets = {"net1": AbstractNet1, "net2": AbstractNet2, "net3": AbstractNet3}
    return abstract_nets[net_name]


class DeepPolyVerifier:
    def __init__(self, net, net_name):
        abstract_net_class = get_anet_class_from_name(net_name)
        self.abstract_net = abstract_net_class(net)
        self.N = 10
        self.gamma = 4

    def verify(self, inputs, eps, true_label) -> bool:
        abstract_input = create_abstract_input_shape(inputs, eps)
        while True:
            final_abstract_shape = self.abstract_net.forward(
                abstract_input, true_label, self.N
            )
            if verifyFinalShape(final_abstract_shape):
                return True
            loss = weightedLoss(final_abstract_shape.lower, self.gamma)
            alphas = self.abstract_net.get_alphas()
            gradient = torch.autograd.grad(loss, alphas)
            # TODO: do we need to detach??
            # input_gradient = [grad.detach() for grad in input_gradient]
            assert gradient.shape == alphas.shape
            self.abstract_net.set_alphas(alphas + gradient)


def verifyFinalShape(final_shape: AbstractShape) -> bool:
    l = final_shape.lower
    return torch.all(torch.greater_equal(l, torch.zeros_like(l))).item()


def main():
    pass


if __name__ == "__main__":
    main()
