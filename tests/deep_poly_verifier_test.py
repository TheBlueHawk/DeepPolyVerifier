import sys

sys.path.append("./code")

from abstract_shape import AbstractShape
from deep_poly_verifier import (
    getFinalLayerWeights,
    verifyFinalShape,
    finalLayerVerification,
)
import torch
from torch import Tensor


def test_getFinalLayerWeights_1():
    target: Tensor = Tensor([[0, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, -1]])
    out = getFinalLayerWeights(0, 3)

    assert torch.allclose(out, target)


def test_getFinalLayerWeights_2():
    target: Tensor = Tensor([[0, -1, 1], [0, 0, 0]])
    out = getFinalLayerWeights(1, 2)
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


def test_finalLayerVerification_1():
    aInput = AbstractShape(
        Tensor([[1, 1], [0, 1]]),
        Tensor([[0, 1], [0, 1]]),
        Tensor([-2, -2]),
        Tensor([2, 2]),
    )

    assert finalLayerVerification(aInput, 0, 2) == True
