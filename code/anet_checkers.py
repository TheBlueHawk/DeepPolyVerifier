import torch

class ANetChecker():
    def __init__(self, net):
        self.layers = net.layers
        self.current_layer = -1

    def reset(self, x):
        self.x = x
        self.current_layer = -1

    def check_next(self, abstract_shape):
        if self.current_layer > -1:
            self.x = self.layers[self.current_layer](self.x)

        self.current_layer += 1
        assert self.x_in_ashape(self.x, abstract_shape)

    def x_in_ashape(self, x, abstract_shape):
        raise NotImplementedError()


class DummyANetChecker(ANetChecker):
    def __init__(self, net):
        pass

    def reset(self, x):
        pass

    def check_next(self, abstract_shape):
        pass

class InclusionANetChecker(ANetChecker):
    def x_in_ashape(self, x, abstract_shape):
        return (torch.all(abstract_shape.lower <= x) and 
                torch.all(x <= abstract_shape.upper))