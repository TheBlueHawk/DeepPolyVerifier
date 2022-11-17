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

def expand_abstract_shape(abstract_shape):
    """Rerepresents a compact relu abstract shape wrt. all neurons in the prevoious layer"""
    
    intercept_greater = torch.zeros_like(abstract_shape.y_greater)
    weights_greater = abstract_shape.y_greater.flatten()
    weights_greater_expanded = torch.diag(weights_greater)
    y_greater = torch.cat([intercept_greater, weights_greater_expanded], axis=1)

    intercept_less = abstract_shape.y_less[:, 0:1]
    weights_less = abstract_shape.y_less[:, 1]
    weights_less_expanded = torch.diag(weights_less)
    y_less = torch.cat([intercept_less, weights_less_expanded], axis=1)

    return AbstractShape(
        y_greater, 
        y_less, 
        abstract_shape.lower, 
        abstract_shape.upper)
    


def create_abstract_input_shape(inputs, eps):
    return AbstractShape(
        y_greater=torch.clamp(inputs-eps, 0, 1),
        y_less=torch.clamp(inputs+eps, 0, 1),
        lower=torch.clamp(inputs-eps, 0, 1),
        upper=torch.clamp(inputs+eps, 0, 1)
    )

def main():
    pass

if __name__ == '__main__':
    main()