import sys

sys.path.append("../code")

from abstract_shape import AbstractShape
from transformers import AbstractLinear, AbstractReLU, AbstractNormalize
import torch


def test_AbstractLinear():
    a_input = AbstractShape(
        torch.tensor([-1, -1]).reshape(-1, 1),
        torch.tensor([1, 1]).reshape(-1, 1),
        torch.tensor([-1, -1]),
        torch.tensor([1, 1]),
    )
    weights = torch.tensor([[0, 1, 2], [-1, -2, 1]])
    a_linear = AbstractLinear(weights)

    out = a_linear.forward(a_input)

    assert torch.allclose(out.lower, torch.tensor([-3, -4]))
    assert torch.allclose(out.upper, torch.tensor([3, 2]))


def test_AbstrctReLU_crossing_1():
    aInput = AbstractShape(
        torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, -1.0]]),
        torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, -1.0]]),
        torch.tensor([-2.0, -2.0]),
        torch.tensor([2.0, 2.0]),
    )
    aReLU = AbstractReLU()

    out = aReLU.forward(aInput)

    print(out.y_greater, out.y_less, out.lower, out.upper)

    assert torch.allclose(out.y_greater, torch.tensor([[0.0], [0.0]]))
    assert torch.allclose(out.y_less, torch.tensor([[1.0, 0.5], [1, 0.5]]))
    assert torch.allclose(out.lower, torch.tensor([0.0, 0.0]))
    assert torch.allclose(out.upper, torch.tensor([2.0, 2.0]))


def test_AbstrctReLU_crossing_2():
    aInput = AbstractShape(
        torch.tensor([[-0.5, 1.0, 1.0], [0.0, 1.0, -1.0]]),
        torch.tensor([[-0.5, 1.0, 1.0], [0.0, 1.0, -1.0]]),
        torch.tensor([-0.5, -2.0]),
        torch.tensor([2.5, 2.0]),
    )
    aReLU = AbstractReLU()

    out = aReLU.forward(aInput)

    print(out.y_greater, out.y_less, out.lower, out.upper)

    assert torch.allclose(out.y_greater, torch.tensor([[0.0], [0.0]]))
    assert torch.allclose(out.y_less, torch.tensor([[5 / 12, 5 / 6], [1, 0.5]]))
    assert torch.allclose(out.lower, torch.tensor([0.0, 0.0]))
    assert torch.allclose(out.upper, torch.tensor([2.5, 2.0]))


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
