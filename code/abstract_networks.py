from transformers import (AbstractNormalize, AbstractFlatten, 
    AbstractLinear, AbstractReLU)

class AbstractNetwork:
    def __init__(
        self,
        abstract_transformers: List[AbstractShape],
    ) -> None:
        ## TODO: sparse representation for large number of unused variables
        self.abstract_transformers = abstract_transformers

    def get_abstract_transformers(self):
        return self.abstract_transformers

class AbstractNet1:
    def __init__(
        self,
        net
    ) -> None:
        self.abstract_transformers = [
            AbstractFlatten(),
            AbstractNormalize(),
            AbstractLinear(),
            AbstractReLU(),
            AbstractLinear()
        ]