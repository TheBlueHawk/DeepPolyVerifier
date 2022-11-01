from tokenize import Double
from turtle import forward
from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from code.representations import AbstractLayer

class AbstractLinear():
    def __init__(
        self,
        weights: Tensor
    ) -> None:
        self.weights = weights
    
    def forward(self, x: AbstractLayer, weights: Tensor):
        return NotImplementedError

class AbstractFlatten():     

    def forward(self, x: AbstractLayer):
        return nn.Flatten(0).forward(x)


class AbstractNormalize():
    def __init__(
        self,
        mean: torch.FloatTensor,
        sigma: torch.FloatTensor,
    ) -> None:
        self.mean = mean
        self.sigma = sigma
    
    def forward(self, x: AbstractLayer):
        return NotImplementedError


class AbstractReLU():
    def __init__(
        self
    ) -> None:
        raise NotImplementedError
    
    def forward(self, x: AbstractLayer):
        return NotImplementedError