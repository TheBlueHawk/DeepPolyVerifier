import sys

sys.path.append("../code")

from transformers import AbstractLinear, AbstractNormalize
from representations import AbstractShape
import torch


def test_AbstractLinear():
    aInput = AbstractShape(
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

def test_AbstractNormalize():
    aInput = AbstractShape(
        torch.FloatTensor([-1, -2]).reshape(-1, 1),
        torch.FloatTensor([1, 3]).reshape(-1, 1),
        torch.FloatTensor([-1, -2]),
        torch.FloatTensor([1, 3]),
    )
    aNorm = AbstractNormalize(1, 2)

    out = aNorm.forward(aInput)

    assert torch.allclose(out.lower, torch.FloatTensor([-1, -1.5]))
    assert torch.allclose(out.upper, torch.FloatTensor([0, 1]))