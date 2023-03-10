import torch
from networks import NormalizedResnet
from resnet import BasicBlock


class ANetChecker:
    def __init__(self, net):
        if isinstance(net, NormalizedResnet):
            # Only goes throught level 0 and 1, not inside the resblock
            self.layers = [net.normalization]
            for l in net.resnet:
                if isinstance(l, torch.nn.Sequential):
                    for subl in l:
                        self.layers.append(subl)
                else:
                    self.layers.append(l)
        elif isinstance(net, torch.nn.Sequential):
            self.layers = [l for l in net]
        else:
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
            raise Exception("Abstract shape doesn't contain the concrete point or has nans")

    def recheck(self, abstract_shape):
        if not self.x_in_ashape(self.x.squeeze(0), abstract_shape):
            raise Exception("Abstract shape doesn't contain the concrete point")

    def x_in_ashape(self, x, abstract_shape):
        raise NotImplementedError()


class DummyANetChecker(ANetChecker):
    def __init__(self, net):
        pass

    def reset(self, x):
        self.x = x

    def check_next(self, abstract_shape):
        pass

    def recheck(self, abstract_shape):
        pass

    def next_path_a_checker(self):
        return DummyANetChecker(None)

    def next_path_b_checker(self):
        return DummyANetChecker(None)


class InclusionANetChecker(ANetChecker):
    def x_in_ashape(self, x, abstract_shape):
        bad_idx_l = abstract_shape.lower > x
        bad_idx_u = abstract_shape.upper < x

        sat1 = torch.all(
            (abstract_shape.lower <= x) | torch.isclose(abstract_shape.lower, x)
        ) and torch.all(
            (x <= abstract_shape.upper) | torch.isclose(abstract_shape.upper, x)
        )
        sat2 = ~(
            (abstract_shape.y_greater is not None and torch.isnan(abstract_shape.y_greater).any()) |
            ((abstract_shape.y_greater is not None and  torch.isnan(abstract_shape.y_less).any()))|
            torch.isnan(abstract_shape.lower).any() |
            torch.isnan(abstract_shape.upper).any()
        )

        if not sat2:
            print("Nans")


        if not sat1:
            pts = 5
            torch.set_printoptions(precision=10)
            print(
                "upper-lower:",
                (
                    abstract_shape.upper[bad_idx_l][:pts]
                    - abstract_shape.lower[bad_idx_l][:pts]
                ),
                sep="\n",
            )
            print(
                "bad lower x < l",
                "x",
                x[bad_idx_l][:pts],
                "l",
                abstract_shape.lower[bad_idx_l][:pts],
                sep="\n",
            )
            print(
                "bad upper x > u",
                "x",
                x[bad_idx_u][:pts],
                "u",
                abstract_shape.upper[bad_idx_u][:pts],
                sep="\n",
            )
            torch.set_printoptions()

        return sat1 and sat2


class ResnetChecker(InclusionANetChecker):
    def __init__(self, net):
        self.path_a_checkers = []
        self.path_b_checkers = []
        self.layers = [net.normalization]
        for l in net.resnet:
            if isinstance(l, torch.nn.Sequential):
                for subl in l:
                    self.layers.append(subl)
                    if isinstance(subl, BasicBlock):
                        if len(subl.path_a) > 1:
                            self.path_a_checkers.append(
                                InclusionANetChecker(subl.path_a[1:])
                            )
                        else:
                            self.path_a_checkers.append(
                                InclusionANetChecker(subl.path_a)
                            )
                        self.path_b_checkers.append(InclusionANetChecker(subl.path_b))
            else:
                self.layers.append(l)

        self.current_layer = -1
        self.current_path_a_checker = -1
        self.current_path_b_checker = -1

    def next_path_a_checker(self):
        self.current_path_a_checker += 1
        return self.path_a_checkers[self.current_path_a_checker]

    def next_path_b_checker(self):
        self.current_path_b_checker += 1
        return self.path_b_checkers[self.current_path_b_checker]


# class BlockCkecker(InclusionANetChecker):
#     def __init__(self, layers):
#         self.layers = layers
#         self.current_layer = -1
