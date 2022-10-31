from tokenize import Double
from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor

class Inequality():
    def __init__(
        self,
        var_pairs: List[Tuple[int, Double]],
        num_vars: int
    ) -> None:
        self.n = num_vars
        self.coeff = torch.tensor([0] * num_vars)
        for index, value in var_pairs:
            self.coeff[index] = value

class Layer():
    def __init__(
        self,
        neurons: List[Inequality],
    ) -> None:
        self.neurons = neurons

class Network():
    def __init__(
        self,
        layers: List[Layer],
    ) -> None:
        self.neurons = layers