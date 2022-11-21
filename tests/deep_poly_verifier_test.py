import sys

sys.path.append("./code")

from abstract_shape import AbstractShape
from abstract_networks import AbstractNetwork
from deep_poly_verifier import (
    verifyFinalShape
)
import torch
from torch import Tensor


def test_buildFinalLayerWeights_1():
    a_net = AbstractNetwork([])
    target: Tensor = Tensor([[0, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, -1]])
    out = a_net.buildFinalLayerWeights(0, 3)

    assert torch.allclose(out, target)


def test_buildFinalLayerWeights_2():
    a_net = AbstractNetwork([])
    target: Tensor = Tensor([[0, -1, 1], [0, 0, 0]])
    out = a_net.buildFinalLayerWeights(1, 2)
    assert torch.allclose(out, target)


def test_verifyFinalShape_1():
    aInput = AbstractShape(
        None,
        None,
        torch.tensor([-0.1, 0, 2]),
        None,
    )

    assert verifyFinalShape(aInput) == False


def test_verifyFinalShape_2():
    aInput = AbstractShape(
        None,
        None,
        torch.tensor([0, 0, 2]),
        None,
    )

    assert verifyFinalShape(aInput) == True
