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
    torch.tensor([1, 1]))
    weights = torch.tensor([[0, 1, 2], [-1, -2, 1]])
    aLinear = AbstractLinear(weights)

    out = aLinear.forward(aInput)
    
    assert torch.allclose(out.lower, torch.tensor([-3, -4]))
    assert torch.allclose(out.upper, torch.tensor([3, 2]))