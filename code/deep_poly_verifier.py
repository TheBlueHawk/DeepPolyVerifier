import torch
from torch import Tensor
import time

from abstract_shape import create_abstract_input_shape, AbstractShape, weightedLoss
from abstract_networks import (
    AbstractNet1,
    AbstractNet10,
    AbstractNet2,
    AbstractNet3,
    AbstractNet4,
    AbstractNet5,
    AbstractNet6,
    AbstractNet7,
    AbstractNet8,
    AbstractNet9,
    AbstractNetwork,
)
from anet_checkers import (
    ANetChecker,
    DummyANetChecker,
    InclusionANetChecker,
    ResnetChecker,
)


def get_anet_class_from_name(net_name) -> AbstractNetwork:
    abstract_nets = {
        "net1": AbstractNet1,
        "net2": AbstractNet2,
        "net3": AbstractNet3,
        "net4": AbstractNet4,
        "net5": AbstractNet5,
        "net6": AbstractNet6,
        "net7": AbstractNet7,
        "net8": AbstractNet8,
        "net9": AbstractNet9,
        "net10": AbstractNet10,
    }
    return abstract_nets[net_name]


# def get_checker_class_from_name(net_name) -> ANetChecker:
#     checkers = {
#         "net1": DummyANetChecker,
#         "net2": DummyANetChecker,
#         "net3": DummyANetChecker,
#         "net4": DummyANetChecker,
#         "net5": DummyANetChecker,
#         "net6": DummyANetChecker,
#         "net7": DummyANetChecker,
#     }
#     return checkers[net_name]


def get_checker_class_from_name(net_name) -> ANetChecker:
    checkers = {
        "net1": DummyANetChecker,
        "net2": DummyANetChecker,
        "net3": DummyANetChecker,
        "net4": DummyANetChecker,
        "net5": DummyANetChecker,
        "net6": DummyANetChecker,
        "net7": DummyANetChecker,
        "net8": DummyANetChecker,
        "net9": DummyANetChecker,
        "net10": DummyANetChecker,
    }
    return checkers[net_name]


class DeepPolyVerifier:
    def __init__(self, net, net_name):
        self.net = net
        checker_class = get_checker_class_from_name(net_name)
        self.checker = checker_class(net)
        abstract_net_class = get_anet_class_from_name(net_name)
        self.abstract_net = abstract_net_class(net, self.checker)
        self.N = 10
        self.gamma = 5
        self.ALPHA_EPOCHS = 10
        self.ALPHA_ITERS = 20
        self.LR = 1.5
        self.WEIGHT_DECAY = 0

    def verify(self, inputs, eps, true_label) -> bool:
        """
        Args:
            inputs: tensor of shape <channels, width, height>

        Returns:
            Boolean
        """
        # torch.autograd.set_detect_anomaly(True)
        abstract_input = create_abstract_input_shape(inputs.squeeze(0), eps)
        self.checker.reset(inputs)

        start_time = time.time()
        for _ in range(self.ALPHA_EPOCHS):
            for _ in range(self.ALPHA_ITERS):
                if time.time() - start_time > 60:
                    return False

                final_abstract_shape = self.abstract_net.forward(
                    abstract_input, true_label, self.N
                )
                # print(f"Max violation:\t {final_abstract_shape.lower.min()}\n")
                if verifyFinalShape(final_abstract_shape):
                    return True

                self.checker.reset(inputs)

                # Do just one step and then recompute the output
                # Alternatively could do multiple step with the existing mapping
                # from input neurons to bounds

                alphas = self.abstract_net.get_alphas()
                # print([(float(a.min()), float(a.max())) for a in alphas])
                optim = torch.optim.AdamW(
                    alphas, lr=self.LR  # , weight_decay=self.WEIGHT_DECAY
                )
                optim.zero_grad()
                loss = weightedLoss(final_abstract_shape.lower, self.gamma)
                loss.backward()
                # print("percentage of nans")
                # print([(a.grad.isnan().sum(), a.grad.size()) for a in alphas])
                for a in alphas:
                    a.grad[a.grad.isnan()] = 0
                optim.step()
                # alphas_clamped = [
                #     a.clamp(0, 1).detach().requires_grad_() for a in alphas
                # ]
                # self.abstract_net.set_alphas(alphas_clamped)

            alphas = self.abstract_net.get_alphas()
            new_alphas = [
                torch.normal(0, 0.2, a.shape, requires_grad=True) for a in alphas
            ]
            self.abstract_net.set_alphas(new_alphas)
        return False


def verifyFinalShape(final_shape: AbstractShape) -> bool:
    l = final_shape.lower
    return torch.all(torch.greater_equal(l, torch.zeros_like(l))).item()


def main():
    pass


if __name__ == "__main__":
    main()
