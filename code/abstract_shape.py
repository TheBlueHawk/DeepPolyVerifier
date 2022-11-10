from tokenize import Double
from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

class AbstractShape:
    def __init__(
        self,
        y_greater: Tensor,  # y >= ...
        y_less: Tensor,  # y <= ...
        lower: Tensor,
        upper: Tensor,
    ) -> None:
        self.y_greater = y_greater
        self.y_less = y_less
        self.upper = upper
        self.lower = lower

    def __str__(self):
        return f"{self.y_greater}\n{self.y_less}\n{self.lower}\n{self.upper}"


def create_abstract_input_shape(inputs, eps):
    return AbstractShape(
        y_greater=torch.clamp(inputs-eps, 0, 1), 
        y_less=torch.clamp(inputs+eps, 0, 1), 
        lower=torch.clamp(inputs-eps, 0, 1), 
        upper=torch.clamp(inputs+eps, 0, 1)
    )