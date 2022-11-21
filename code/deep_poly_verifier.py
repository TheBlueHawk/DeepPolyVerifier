import torch
from torch import Tensor

from abstract_shape import create_abstract_input_shape, AbstractShape
from abstract_networks import AbstractNet1, AbstractNet2, AbstractNet3
from transformers import AbstractLinear


def get_anet_class_from_name(net_name):
    abstract_nets = {
        "net1": AbstractNet1,
        "net2": AbstractNet2,
        "net3": AbstractNet3
        }
    return abstract_nets[net_name]


class DeepPolyVerifier:
    def __init__(self, net, net_name, use_final_layer=False):
        abstract_net_class = get_anet_class_from_name(net_name)
        self.abstract_net = abstract_net_class(net)
        self.N = 10

    def verify(self, inputs, eps, true_label):
        abstract_input = create_abstract_input_shape(inputs, eps)
        final_abstract_shape = self.abstract_net.forward(abstract_input, true_label, self.N)
        return verifyFinalShape(final_abstract_shape)


def verifyFinalShape(final_shape: AbstractShape) -> bool:
    l = final_shape.lower
    return torch.all(torch.greater_equal(l, torch.zeros_like(l))).item()


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
