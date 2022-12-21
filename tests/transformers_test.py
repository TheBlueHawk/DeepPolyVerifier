import sys
import pytest
from torch import Tensor
from abstract_networks import create_abstract_id_conv

sys.path.append("./code")

from abstract_shape import (
    AbstractShape,
    ConvAbstractShape,
    create_abstract_input_shape,
    LinearAbstractShape,
)
from transformers import (
    AbstractBatchNorm,
    AbstractLinear,
    AbstractReLU,
    AbstractNormalize,
    AbstractConvolution,
)
import torch


def test_AbstractLinear():
    a_input = AbstractShape(
        torch.tensor([-1, -1]).reshape(-1, 1),
        torch.tensor([1, 1]).reshape(-1, 1),
        torch.tensor([-1, -1]),
        torch.tensor([1, 1]),
    )
    weights = torch.tensor([[0, 1, 2], [-1, -2, 1]])
    a_linear = AbstractLinear(weights)

    out = a_linear.forward(a_input)

    assert torch.allclose(out.lower, torch.tensor([-3, -4]))
    assert torch.allclose(out.upper, torch.tensor([3, 2]))


def test_AbstrctReLU_crossing_1():
    aInput = LinearAbstractShape(
        torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, -1.0]]),
        torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, -1.0]]),
        torch.tensor([-2.0, -2.0]),
        torch.tensor([2.0, 2.0]),
    )
    aReLU = AbstractReLU(alpha_init="zeros")

    out = aReLU.forward(aInput)

    assert torch.allclose(out.y_greater, torch.tensor([[0.0], [0.0]]))
    assert torch.allclose(out.y_less, torch.tensor([[1.0, 0.5], [1, 0.5]]))
    assert torch.allclose(out.lower, torch.tensor([0.0, 0.0]))
    assert torch.allclose(out.upper, torch.tensor([2.0, 2.0]))


def test_AbstrctReLU_crossing_2():
    aInput = LinearAbstractShape(
        torch.tensor([[-0.5, 1.0, 1.0], [0.0, 1.0, -1.0]]),
        torch.tensor([[-0.5, 1.0, 1.0], [0.0, 1.0, -1.0]]),
        torch.tensor([-0.5, -2.0]),
        torch.tensor([2.5, 2.0]),
    )
    aReLU = AbstractReLU(alpha_init="zeros")

    out = aReLU.forward(aInput)

    print(out.y_greater.shape)
    print(out.y_greater)
    assert torch.allclose(out.y_greater, torch.tensor([[0.0], [0.0]]))
    assert torch.allclose(out.y_less, torch.tensor([[5 / 12, 5 / 6], [1, 0.5]]))
    assert torch.allclose(out.lower, torch.tensor([0.0, 0.0]))
    assert torch.allclose(out.upper, torch.tensor([2.5, 2.0]))


def test_AbstractReLu_mixed_1():
    aInput = LinearAbstractShape(
        torch.tensor([[0.0, 1.0, 1.0, 1, 1], [0.0, 1.0, -1.0, 1, 1]]),
        torch.tensor([[0.0, 1.0, 1.0, 1, 1], [0.0, 1.0, -1.0, 1, 1]]),
        torch.tensor([-2.0, -2.0, 1, -2]),
        torch.tensor([2.0, 2.0, 2, -1]),
    )
    aReLU = AbstractReLU(alpha_init="zeros")

    out = aReLU.forward(aInput)

    assert torch.allclose(out.y_greater, torch.tensor([[0.0], [0.0], [1], [0]]))
    assert torch.allclose(
        out.y_less, torch.tensor([[1.0, 0.5], [1, 0.5], [0, 1], [0, 0]])
    )
    assert torch.allclose(out.lower, torch.tensor([0.0, 0.0, 1, 0]))
    assert torch.allclose(out.upper, torch.tensor([2.0, 2.0, 2, 0]))


def test_AbstractNormalize():
    aInput = AbstractShape(
        torch.FloatTensor([-1, -2]).reshape(-1, 1),
        torch.FloatTensor([1, 3]).reshape(-1, 1),
        torch.FloatTensor([-1, -2]),
        torch.FloatTensor([1, 3]),
    )
    aNorm = AbstractNormalize(1, 2)

    out = aNorm.forward(aInput)

    assert torch.allclose(out.lower, torch.FloatTensor([-1, -1.5]))
    assert torch.allclose(out.upper, torch.FloatTensor([0, 1]))


def test_AbstractConvolution_shapes():
    layer = torch.nn.Conv2d(1, 16, (4, 4), 2, 1)
    a_layer = AbstractConvolution(layer)
    img = torch.ones((1, 28, 28))
    a_shape = create_abstract_input_shape(img, 0.1)

    out = a_layer.forward(a_shape)

    assert out.y_greater.shape == (16, 14, 14, 17)
    assert out.y_less.shape == (16, 14, 14, 17)
    assert out.upper.shape == (16, 14, 14)
    assert out.lower.shape == (16, 14, 14)


def test_AbstractConvolution_1():
    layer = torch.nn.Conv2d(1, 16, 4, 2, 1)
    c_out, c_in, h, w = 2, 2, 3, 3
    kh, kw = 2, 2
    padding = 0
    stride = 1
    kernel = 1 + torch.arange(c_out * c_in * kh * kw).reshape(c_out, c_in, kh, kw)
    bias = torch.tensor([2.0, 1])

    img = torch.zeros(c_in, h, w)
    img[:, 1, 1] += 1
    img[0, 0, :] += 1
    img[1, 1, :] += 1
    a_transformer = AbstractConvolution(kernel, bias, stride, padding)
    a_shape = create_abstract_input_shape(img, 0, bounds=(-10, 10))

    out = a_transformer.forward(a_shape)

    assert torch.allclose(
        out.y_greater,
        torch.tensor(
            [[2.0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 9, 10, 11, 12, 13, 14, 15, 16]]
        )
        .unsqueeze(1)
        .unsqueeze(1)
        .repeat(1, 2, 2, 1),
    )
    assert torch.allclose(out.y_less, out.y_greater)
    assert torch.allclose(
        out.upper,
        torch.tensor(
            [
                [[32.0, 30], [21, 19]],
                [[79, 77], [52, 50]],
            ]
        ),
    )
    assert torch.allclose(out.lower, out.upper)


def test_AbstractConvolution_2():
    layer = torch.nn.Conv2d(1, 16, 4, 2, 1)
    c_out, c_in, h, w = 2, 2, 3, 3
    kh, kw = 2, 2
    padding = 0
    stride = 1
    kernel = torch.concat(
        [
            torch.arange(-5, 3).reshape(1, 2, 2, 2),
            torch.arange(-2, 6).reshape(1, 2, 2, 2),
        ],
        axis=0,
    )
    bias = torch.tensor([2.0, 1])

    img = torch.zeros(c_in, h, w)
    img[:, 1, 1] += 1
    img[0, 0, :] += 1
    img[1, 1, :] += 1
    a_transformer = AbstractConvolution(kernel, bias, stride, padding)
    a_shape = create_abstract_input_shape(img, 1, bounds=(-10, 10))

    out = a_transformer.forward(a_shape)

    assert torch.allclose(
        out.upper,
        torch.tensor(
            [
                [[14.0, 12], [15, 13]],
                [[31, 29], [26, 24]],
            ]
        ),
    )

    assert torch.allclose(
        out.lower,
        torch.tensor(
            [
                [[-22.0, -24], [-21, -23]],
                [[-5, -7], [-10, -12]],
            ]
        ),
    )


def test_AbstractNormalizer():
    y_less_up = Tensor([0, 2, 2, 4, -1, -2, 2, 1]).reshape(2, 2, 2)
    y_eq = (
        Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(2, 1, 1, 5).expand(2, 2, 2, 5)
    )
    input_shape = ConvAbstractShape(y_eq, y_eq, y_less_up, y_less_up, 1, 4, 2, 0, 2)

    num_features = 2
    running_mean = Tensor([0, 1])
    running_var = Tensor([4, 1])
    gamma = Tensor([1, 2])
    beta = Tensor([1, -1])
    eps = 0
    bn = AbstractBatchNorm(num_features, running_mean, running_var, gamma, beta, eps)

    tgt_y_less_up = Tensor([1, 2, 2, 3, -5, -7, 1, -1]).reshape(2, 2, 2)
    tgt_y_eq = (
        Tensor([1.5, 2, 2.5, 3, 3.5, 9, 11, 13, 15, 17])
        .reshape(2, 1, 1, 5)
        .expand(2, 2, 2, 5)
    )
    tgt_shape = ConvAbstractShape(
        tgt_y_eq, tgt_y_eq, tgt_y_less_up, tgt_y_less_up, 1, 4, 2, 0, 2
    )

    out_shape = bn.forward(input_shape)

    assert torch.allclose(tgt_shape.lower, out_shape.lower)
    assert torch.allclose(tgt_shape.y_less, out_shape.y_less)


def test_create_abstract_id_conv():
    c_in = 3
    n_in = 16
    img = torch.zeros(c_in, n_in, n_in)
    img[:, 1, 1] += 1
    img[0, 0, :] += 1
    img[1, 1, :] += 1

    a_shape = create_abstract_input_shape(img, 0, bounds=(-10, 10))

    id_conv = create_abstract_id_conv(3, 3)


def main():
    test_AbstractNormalizer()


if __name__ == "__main__":
    main()
