from tokenize import Double
from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

class Neuron():
    def __init__(
        self,
        var_pairs: List[Tuple[int, Double]],
        num_neurons_in_network: int
    ) -> None:
        ## TODO: sparse representation for large number of unused variables
        self.coeff = torch.tensor([0] * num_neurons_in_network)
        for index, value in var_pairs:
            self.coeff[index] = value

class Layer():
    
    def __init__(
        self,
        neurons: List[Neuron],
    ) -> None:
        ## TODO: sparse representation for large number of unused variables
        # self.neurons = pad_sequence(neurons).T ?
        self.neurons = torch.stack(neurons, dim=0)

class Network():
    def __init__(
        self,
        layers: List[Layer],
    ) -> None:
        ## TODO: sparse representation for large number of unused variables
        self.layers = torch.stack(layers, dim=0)