from audioop import cross
from tokenize import Double
from turtle import forward
from typing import List, Tuple
from xmlrpc.client import Boolean
import torch
import torch.nn as nn
from torch import Tensor

from representations import AbstractShape


class AbstractLinear:
    def __init__(self, weights: Tensor) -> None:
        self.weights = weights

    def forward(self, x: AbstractShape):
        l = torch.cat([torch.tensor([1]), x.lower])
        L = l.repeat((self.weights.shape[0], 1))
        u = torch.cat([torch.tensor([1]), x.upper])
        U = u.repeat((self.weights.shape[0], 1))

        positive_weights = self.weights > 0
        LU_minor = torch.where(positive_weights, L, U)
        LU_greater = torch.where(positive_weights, U, L)

        return AbstractShape(
            y_greater=self.weights,
            y_less=self.weights,
            upper=torch.sum(LU_greater * self.weights, dim=1),
            lower=torch.sum(LU_minor * self.weights, dim=1),
        )


class AbstractFlatten:
    def forward(self, x: AbstractShape):
        return nn.Flatten(0).forward(x)


class AbstractNormalize:
    def __init__(
        self,
        mean: torch.FloatTensor,
        sigma: torch.FloatTensor,
    ) -> None:
        self.mean = mean
        self.sigma = sigma

    def forward(self, x: AbstractShape):
        y_greater_one_neur = torch.tensor([-self.mean / self.sigma, 1 / self.sigma])
        y_greater = y_greater_one_neur.repeat((x.lower.shape[0], 1))

        y_less_one_neur = torch.tensor([-self.mean / self.sigma, 1 / self.sigma])
        y_less = y_less_one_neur.repeat((x.lower.shape[0], 1))

        lower = (x.lower - self.mean) / self.sigma
        upper = (x.upper - self.mean) / self.sigma

        return AbstractShape(
            y_greater=y_greater,
            y_less=y_less,
            upper=upper,
            lower=lower,
        )


class AbstractReLU:
    def __init__(self) -> None:
        return

    def forward(self, x: AbstractShape):
        u_i = x.upper
        l_i = x.lower
        a_less_i = x.y_less
        a_greater_i = x.y_greater
        # strictly negative: zero out
        stricly_negative = u_i <= 0
        u_j = torch.where(stricly_negative, torch.zeros_like(u_i), u_i)
        l_j = torch.where(stricly_negative, torch.zeros_like(l_i), l_i)
        a_less_j = torch.where(stricly_negative, torch.zeros_like(u_i), a_less_i)
        a_greater_j = torch.where(stricly_negative, torch.zeros_like(u_i), a_greater_i)

        # strictly positive: return unchanged
        stricly_positive = l_i >= 0
        u_j = torch.where(stricly_positive, u_i, u_j)
        l_j = torch.where(stricly_positive, l_i, l_j)
        a_less_j = torch.where(stricly_positive, a_less_i, a_less_j)
        a_greater_j = torch.where(stricly_positive, a_greater_i, a_greater_j)

        # crossing: keep upperbound, lowerbound at zero, greater_than zero, less than slope
        crossing = (l_i <= 0) & (u_i >= 0)
        slope = u_i / (u_i - l_i)
        u_j = torch.where(crossing, u_i, u_j)
        l_j = torch.where(crossing, torch.zeros_like(l_i), l_j)
        a_less_j = torch.where(crossing, torch.zeros_like(u_i), a_less_j)
        a_greater_j = torch.where(crossing, slope * (a_greater_i - l_i), a_greater_j)

        return AbstractShape(a_greater_j, a_less_j, u_j, l_j)


def main():
    aInput = AbstractShape(
        torch.tensor([-1, -2]).reshape(-1, 1),
        torch.tensor([1, 3]).reshape(-1, 1),
        torch.tensor([-1, -2]),
        torch.tensor([1, 3]),
    )
    aNorm = AbstractNormalize(1, 2)
    print(aNorm.forward(aInput))


if __name__ == "__main__":
    main()
