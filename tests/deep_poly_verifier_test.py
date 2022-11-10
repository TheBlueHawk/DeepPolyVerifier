import sys

sys.path.append("../code")

from deep_poly_verifier import getFinalLayerWeights
import torch
from torch import Tensor


def test_getFinalLayerWeights_1():
    target: Tensor = Tensor([[1, 0, 0], [1, -1, 0], [1, 0, -1]])
    out = getFinalLayerWeights(0, 3)

    assert torch.allclose(out, target)


def test_getFinalLayerWeights_2():
    target: Tensor = Tensor([[-1, 1], [0, 1]])
    out = getFinalLayerWeights(1, 2)
    assert torch.allclose(out, target)
