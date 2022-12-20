import torch


class ANetChecker:
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

        if not self.x_in_ashape(self.x.squeeze(0), abstract_shape):
            raise Exception("Abstract shape doesn't contain the concrete point")

    def recheck(self, abstract_shape):
        if not self.x_in_ashape(self.x.squeeze(0), abstract_shape):
            raise Exception("Abstract shape doesn't contain the concrete point")

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
        bad_idx_l = abstract_shape.lower > x
        bad_idx_u = abstract_shape.upper < x

        ret = torch.all(
            (abstract_shape.lower <= x) | torch.isclose(abstract_shape.lower, x)
        ) and torch.all(
            (x <= abstract_shape.upper) | torch.isclose(abstract_shape.upper, x)
        )

        if not ret:
            pts = 5
            torch.set_printoptions(precision=10)
            print(
                "intervals:",
                (
                    abstract_shape.upper[bad_idx_l][:pts]
                    - abstract_shape.lower[bad_idx_l][:pts]
                ),
                sep="\n",
            )
            print(
                "bad lower x < l",
                x[bad_idx_l][:pts],
                abstract_shape.lower[bad_idx_l][:pts],
                sep="\n",
            )
            print(
                "bad upper x > u",
                x[bad_idx_u][:pts],
                abstract_shape.upper[bad_idx_u][:pts],
                sep="\n",
            )
            torch.set_printoptions()

        return ret
