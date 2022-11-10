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

    def forward(self, x: AbstractLayer):
        l = torch.cat([torch.tensor([1]), x.lower])
        L = l.repeat((self.weights.shape[0], 1))
        u = torch.cat([torch.tensor([1]), x.upper])
        U = u.repeat((self.weights.shape[0], 1))

        positive_weights = self.weights > 0
        LU_minor = torch.where(positive_weights, L, U)
        LU_greater = torch.where(positive_weights, U, L)

        return AbstractLayer(
            y_greater=self.weights,
            y_less=self.weights,
            upper=torch.sum(LU_greater * self.weights, dim=1),
            lower=torch.sum(LU_minor * self.weights, dim=1),
        )


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
        y_greater_one_neur = torch.tensor([-self.mean / self.sigma, 1 / self.sigma])
        y_greater = y_greater_one_neur.repeat((x.lower.shape[0], 1))

        y_less_one_neur = torch.tensor([-self.mean / self.sigma, 1 / self.sigma])
        y_less = y_less_one_neur.repeat((x.lower.shape[0], 1))

        lower = (x.lower - self.mean) / self.sigma
        upper = (x.upper - self.mean) / self.sigma

        return AbstractLayer(
            y_greater=y_greater,
            y_less=y_less,
            upper=upper,
            lower=lower,
        )


class AbstractReLU:
    def __init__(self) -> None:
        pass

    def forward(self, x: AbstractLayer):
        # Given u_i.shape = [1,n]
        # output AbstrcatLayer shapes:
        #
        # u_j.shape = l_j.shape = [1,n]
        # a_greater_j = [1,n] (list of alphas, now all alphas = 0)
        # a_less_j = [2,n] (list of linear coeff, [b,a]: b + ax)
        u_i = x.upper
        l_i = x.lower
        zero_ones = torch.stack((torch.zeros_like(u_i), torch.ones_like(u_i)), dim=1)
        # zero_ones = torch.stack((torch.zeros_like(u_i), torch.ones_like(u_i)), dim=0)
        zero_zeros = torch.zeros_like(zero_ones)
        ones = torch.ones_like(u_i)
        zeros = torch.zeros_like(u_i)
        # TODO set alpha as a param
        alpha = 0

        # strictly negative: zero out
        stricly_negative = u_i <= 0
        u_j = torch.where(stricly_negative, zeros, u_i)
        l_j = torch.where(stricly_negative, zeros, l_i)
        a_less_j = torch.where(stricly_negative, zero_zeros, zero_zeros)
        a_greater_j = torch.where(stricly_negative, zeros, zeros)

        # strictly positive: unchanged l,u; set y=x
        stricly_positive = l_i >= 0
        u_j = torch.where(stricly_positive, u_i, u_j)
        l_j = torch.where(stricly_positive, l_i, l_j)
        a_less_j = torch.where(stricly_positive, zero_ones, a_less_j)
        a_greater_j = torch.where(stricly_positive, ones, a_greater_j)

        # crossing: keep upperbound, lowerbound at zero, greater_than zero, less than slope
        crossing = (l_i <= 0) & (u_i >= 0)
        slope = u_i / (u_i - l_i)
        print("slope", u_i, l_i, slope)
        u_j = torch.where(crossing, u_i, u_j)
        l_j = torch.where(crossing, torch.zeros_like(l_i), l_j)
        # print(a_less_j, slope, torch.stack((-1 * l_i, ones)))
        lin_constr = torch.stack((-1 * l_i, ones), dim=1)
        lin_constr = (slope * lin_constr.T).T
        a_less_j = torch.where(crossing, lin_constr, a_less_j)
        a_greater_j = torch.where(crossing, alpha * ones, a_greater_j)

        return AbstractLayer(a_greater_j, a_less_j, l_j, u_j)


# # To be used for backsubstitution
# class AbstractReLU:
#     def __init__(self) -> None:
#         pass

#     def forward(self, x: AbstractLayer):
#         u_i = x.upper
#         l_i = x.lower
#         a_less_i = x.y_less
#         a_greater_i = x.y_greater
#         # strictly negative: zero out
#         stricly_negative = u_i <= 0
#         u_j = torch.where(stricly_negative, torch.zeros_like(u_i), u_i)
#         l_j = torch.where(stricly_negative, torch.zeros_like(l_i), l_i)
#         a_less_j = torch.where(stricly_negative, torch.zeros_like(u_i), a_less_i)
#         a_greater_j = torch.where(stricly_negative, torch.zeros_like(u_i), a_greater_i)

#         # strictly positive: return unchanged
#         stricly_positive = l_i >= 0
#         u_j = torch.where(stricly_positive, u_i, u_j)
#         l_j = torch.where(stricly_positive, l_i, l_j)
#         a_less_j = torch.where(stricly_positive, a_less_i, a_less_j)
#         a_greater_j = torch.where(stricly_positive, a_greater_i, a_greater_j)

#         # crossing: keep upperbound, lowerbound at zero, greater_than zero, less than slope
#         crossing = (l_i <= 0) & (u_i >= 0)
#         slope = u_i / (u_i - l_i)
#         u_j = torch.where(crossing, u_i, u_j)
#         l_j = torch.where(crossing, torch.zeros_like(l_i), l_j)
#         a_less_j = torch.where(crossing, slope * (a_greater_i - l_i), a_greater_j)
#         a_greater_j = torch.where(crossing, torch.zeros_like(u_i), a_less_j)
#         #a_greater_j = torch.where(crossing, alpha * a_greater_i, a_greater_j)

#         return AbstractLayer(a_greater_j, a_less_j, u_j, l_j)


def main():
    aInput = AbstractLayer(
        torch.tensor([[-0.5, 1.0, 1.0], [0.0, 1.0, -1.0]]),
        torch.tensor([[-0.5, 1.0, 1.0], [0.0, 1.0, -1.0]]),
        torch.tensor([-0.5, -2.0]),
        torch.tensor([2.5, 2.0]),
    )
    aReLU = AbstractReLU()

    print(aReLU.forward(aInput))


if __name__ == "__main__":
    main()
