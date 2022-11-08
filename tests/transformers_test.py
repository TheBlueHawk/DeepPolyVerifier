import sys

sys.path.append("../code")

from transformers import AbstractLinear
from representations import AbstractLayer
import torch


def test_AbstractLinear():
    aInput = AbstractLayer(
        torch.tensor([-1, -1]).reshape(-1, 1),
        torch.tensor([1, 1]).reshape(-1, 1),
        torch.tensor([-1, -1]),
        torch.tensor([1, 1]),
    )
    weights = torch.tensor([[0, 1, 2], [-1, -2, 1]])
    aLinear = AbstractLinear(weights)

    out = aLinear.forward(aInput)

    assert torch.allclose(out.lower, torch.tensor([-3, -4]))
    assert torch.allclose(out.upper, torch.tensor([3, 2]))


# def test_AbstrctReLU():
#     aInput = AbstractLayer(
#         torch.tensor([[1, 1],[1,-1]]),
#         torch.tensor([[1, 1],[1,-1]]),
#         torch.tensor([-2,-2]),
#         torch.tensor([2,2]))
#     aReLU = AbstractReLU()

#     out = aReLU.forward(aInput)

#     assert torch.allclose(out.y_greater, torch.tensor([0, 0]))
#     assert torch.allclose(out.y_less, torch.tensor([2, 2]))
#     assert torch.allclose(out.lower, torch.tensor([0, 0]))
#     assert torch.allclose(out.upper, torch.tensor([2, 2]))
