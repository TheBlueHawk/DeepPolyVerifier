from audioop import cross
from tokenize import Double
from turtle import forward
from typing import List, Tuple
from xmlrpc.client import Boolean
import torch
import torch.nn as nn
from torch import Tensor

from representations import AbstractLayer


class AbstractLinear:
    def __init__(self, weights: Tensor) -> None:
        self.weights = weights

    def forward(self, x: AbstractLayer, weights: Tensor):
        return NotImplementedError


class AbstractFlatten:
    def forward(self, x: AbstractLayer):
        return nn.Flatten(0).forward(x)


class AbstractNormalize:
    def __init__(
        self,
        mean: torch.FloatTensor,
        sigma: torch.FloatTensor,
    ) -> None:
        self.mean = mean
        self.sigma = sigma

    def forward(self, x: AbstractLayer):
        return NotImplementedError


class AbstractReLU:
    def __init__(self) -> None:
        raise NotImplementedError

    def forward(self, x: AbstractLayer):
        u_i = x.upper
        l_i = x.lower
        a_greater_i = x.weights_greater_lin_comb
        a_minor_i = x.weights_minor_lin_comb
        # strictly negative: zero out
        stricly_negative = u_i <= 0
        u_j = torch.where(stricly_negative, torch.zeros_like(u_i), u_i)
        l_j = torch.where(stricly_negative, torch.zeros_like(l_i), l_i)
        a_greater_j = torch.where(stricly_negative, torch.zeros_like(u_i), a_greater_i)
        a_minor_j = torch.where(stricly_negative, torch.zeros_like(u_i), a_minor_i)

        # strictly positive: return unchanged
        # the following lines have no effect, there are here just for clarity
        stricly_positive = l_i >= 0
        u_j = torch.where(stricly_positive, u_i, u_i)
        l_j = torch.where(stricly_positive, l_i, l_i)
        a_greater_j = torch.where(stricly_positive, a_greater_i, a_greater_i)
        a_minor_j = torch.where(stricly_positive, a_minor_i, a_minor_i)

        # crossing: keep upperbound, lowerbound at zero, greater_than zero, less than slope
        crossing = (l_i <= 0) & (u_i >= 0)
        slope = u_i / (u_i - l_i)
        u_j = torch.where(crossing, u_i, u_i)
        l_j = torch.where(crossing, torch.zeros_like(l_i), l_i)
        a_greater_j = torch.where(crossing, torch.zeros_like(u_i), a_greater_i)
        a_minor_j = torch.where(crossing, slope * (a_minor_i - l_i), a_minor_i)

        return AbstractLayer(a_minor_j, a_greater_j, u_j, l_j)
