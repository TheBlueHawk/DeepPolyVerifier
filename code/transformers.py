from audioop import cross
from tokenize import Double
from turtle import forward
from typing import List, Tuple
from xmlrpc.client import Boolean
import torch
import torch.nn as nn
from torch import Tensor

from abstract_shape import AbstractShape


class AbstractLinear:
    def __init__(self, *args) -> None:
        if isinstance(args[0], nn.Linear):
            self._init_from_layer(args[0])

        elif isinstance(args[0], torch.Tensor):
            self._init_from_tensor(args[0])

        else:
            raise Exception(
                "Invalid arguments passed to the initializer of AbstractLinear"
            )

    def _init_from_tensor(self, weights):
        self.weights = weights.detach()

    def _init_from_layer(self, layer):
        self.weights = torch.cat(
            [layer.bias.data.detach().unsqueeze(1), layer.weight.data.detach()], axis=1
        )

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
        return AbstractShape(
            x.y_greater.reshape(-1, 1),
            x.y_less.reshape(-1, 1),
            x.lower.flatten(),
            x.upper.flatten(),
        )


class AbstractNormalize:
    def __init__(self, *args) -> None:
        if len(args) == 2:
            self._init_from_values(args[0], args[1])
        elif len(args) == 1:
            self._init_from_layer(args[0])

    def _init_from_values(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def _init_from_layer(self, layer):
        self.mean = layer.mean.flatten()
        self.sigma = layer.sigma.flatten()

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
        pass

    def forward(self, x: AbstractShape):
        # Given u_i.shape = [1,n]
        # output AbstrcatLayer shapes:
        #
        # u_j.shape = l_j.shape = [1,n]
        # a_greater_j = [1,n] (list of alphas, now all alphas = 0)
        # a_less_j = [2,n] (list of linear coeff, [b,a]: b + ax)
        u_i = x.upper
        l_i = x.lower
        zero_ones = torch.stack((torch.zeros_like(u_i), torch.ones_like(u_i)), dim=0)
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
        lin_constr = torch.stack((-1 * l_i, ones), dim=0)
        lin_constr = slope * lin_constr
        a_less_j = torch.where(crossing, lin_constr, a_less_j)
        a_greater_j = torch.where(crossing, alpha * ones, a_greater_j)

        return AbstractShape(a_greater_j, a_less_j, l_j, u_j)


# # To be used for backsubstitution
# class AbstractReLU:
#     def __init__(self) -> None:
#         pass

#     def forward(self, x: AbstractShape):
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

#         return AbstractShape(a_greater_j, a_less_j, u_j, l_j)


def main():
    weights = torch.tensor([[0, 1, 2], [-1, -2, 1]])
    linear = nn.Linear(5, 5)
    AbstractLinear(weights)
    AbstractLinear(linear)
    AbstractLinear(2)


if __name__ == "__main__":
    main()
