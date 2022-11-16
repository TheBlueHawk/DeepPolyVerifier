from abstract_shape import AbstractShape
from typing import List
from transformers import (
    AbstractNormalize,
    AbstractFlatten,
    AbstractLinear,
    AbstractReLU,
)


class AbstractNetwork:
    def __init__(
        self,
        abstract_transformers: List,
    ) -> None:
        ## TODO: sparse representation for large number of unused variables
        self.abstract_transformers = abstract_transformers

    def get_abstract_transformers(self):
        return self.abstract_transformers


class AbstractNet1(AbstractNetwork):
    def __init__(self, net) -> None:

        super().__init__(
            [
                AbstractFlatten(),
                AbstractNormalize(net.layers[0]),
                AbstractLinear(net.layers[2]),
                AbstractReLU(),
                AbstractLinear(net.layers[4]),
            ]
        )
