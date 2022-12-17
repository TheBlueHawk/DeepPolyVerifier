import torch
import sys
import pytest

from torch import Tensor

sys.path.append("./code")

from abstract_shape import (
    ConvAbstractShape,
    ReluAbstractShape,
    AbstractShape,
    buildConstraints3DMatrix,
    LinearAbstractShape,
    weightedLoss,
    create_abstract_input_shape
)
from abstract_networks import AbstractNetwork, recompute_bounds
from transformers import AbstractReLU, AbstractLinear, AbstractConvolution


def test_expand_abstract_shape():
    abstract_shape = ReluAbstractShape(
        torch.tensor([[1], [2]]),
        torch.tensor([[3, 4], [5, 6]]),
        torch.tensor([7, 8]),
        torch.tensor([9, 10]),
    )

    out = abstract_shape.expand()

    assert torch.allclose(out.y_greater, torch.tensor([[0, 1, 0], [0, 0, 2]]))
    assert torch.allclose(out.y_less, torch.tensor([[3, 4, 0], [5, 0, 6]]))
    assert torch.allclose(out.lower, torch.tensor([7, 8]))
    assert torch.allclose(out.upper, torch.tensor([9, 10]))


def test_alinear_backsub():
    cur_shape = LinearAbstractShape(
        torch.tensor([[1.0, -1, 1, -1], [-1.0, 1, -1, 1]]),
        torch.tensor([[-1.0, 1, -1, 1], [1.0, -1, 1, -1]]),
        torch.ones(2),
        -torch.ones(2),
    )
    prev_shape = LinearAbstractShape(
        2 * torch.ones(3, 5, dtype=torch.float32),
        torch.stack(
            [torch.zeros(5, dtype=torch.float32), torch.ones(5), -torch.ones(5)]
        ),
        torch.ones(3),
        torch.ones(3),
    )

    new_shape = cur_shape.backsub(prev_shape)
    assert torch.allclose(
        new_shape.y_greater, torch.tensor([[4.0, 3, 3, 3, 3], [2.0, 3, 3, 3, 3]])
    )
    assert torch.allclose(
        new_shape.y_less, torch.tensor([[-4.0, -3, -3, -3, -3], [-2.0, -3, -3, -3, -3]])
    )


def test_buildConstraints3DMatrix_1():
    cur_abstract_shape = AbstractShape(
        torch.tensor([[1.0, -2, -1, 2], [-2, 1, 2, -1]]),
        torch.tensor([[1.0, 1, 1, 1], [1, 1, 1, 1]]),
        torch.tensor([9.0, 10]),
        torch.tensor([11.0, 12]),
    )

    prev_abstract_shape = AbstractShape(
        torch.tensor([[1.0, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]),
        torch.tensor(
            [[10.0, 20, 30, 40, 50], [60, 70, 80, 90, 100], [110, 120, 130, 140, 150]]
        ),
        torch.tensor([9.0, 10, 11]),
        torch.tensor([11.0, 12, 13]),
    )

    cube = buildConstraints3DMatrix(
        cur_weights=cur_abstract_shape.y_greater,
        fst_choice=prev_abstract_shape.y_greater,
        snd_choice=prev_abstract_shape.y_less
    )

    tgt_cube = Tensor(
        [
            [
                [10.0, 20, 30, 40, 50],
                [60.0, 70, 80, 90, 100],
                [11.0, 12, 13, 14, 15],
            ],
            [
                [1.0, 2, 3, 4, 5],
                [6.0, 7, 8, 9, 10],
                [110.0, 120, 130, 140, 150],
            ],
        ],
    )

    assert torch.allclose(cube, tgt_cube)


def test_weightedLoss_1():
    out = Tensor([1, 0, 2, -0.1, -0.5])
    gamma = 10.0
    tgt = Tensor([3.0])
    loss = weightedLoss(out, gamma)
    assert torch.allclose(loss, tgt)


def test_aconv_backsub_conv_1():
    curr_eq = Tensor([-1, -1, 0, 1, 0]).reshape(1, 1, 1, 5)
    curr_shape = ConvAbstractShape(
        curr_eq, curr_eq, None, None, c_in=1, n_in=None, k=2, padding=0, stride=1
    )
    prev_eq = Tensor([1, 1, 0, 0, 1]).repeat(1, 2, 2, 1)
    prev_shape = ConvAbstractShape(
        prev_eq, prev_eq, None, None, c_in=1, n_in=None, k=2, padding=0, stride=1
    )

    out_shape = curr_shape.backsub_conv(prev_shape)
    # print(out_shape.y_greater.shape)

    tgt_eq = Tensor([-1, -1, 0, 0, 1, -1, 0, 0, 1, 0]).reshape(1, 1, 1, 10)
    tgt_shape = ConvAbstractShape(
        tgt_eq, tgt_eq, None, None, c_in=None, n_in=None, padding=None, k=None, stride=None
    )
    # print(tgt_shape.y_greater.shape)

    # print(out_shape.y_greater)
    # print(tgt_shape.y_greater)

    assert torch.allclose(out_shape.y_greater, tgt_shape.y_greater)


def test_aconv_backsub_conv_2():
    curr_eq = Tensor(
        [
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 1, 1],
            [-1, -1, 0, 0, 0, 0, 0, 0, 1],
        ]
    ).reshape(3, 1, 1, 9)
    curr_shape = ConvAbstractShape(
        curr_eq, curr_eq, None, None, c_in=2, k=2, n_in=None, padding=0, stride=1
    )
    prev_eq = (
        Tensor(
            [
                [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, -1, 0],
                [-1, 0, -1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            ]
        )
        .reshape(2, 1, 1, 13)
        .repeat(1, 2, 2, 1)
    )
    prev_shape = ConvAbstractShape(
        prev_eq, prev_eq, None, None, c_in=3, n_in=None, k=2, padding=0, stride=1
    )

    out_shape = curr_shape.backsub_conv(prev_shape)
    # print(out_shape.y_greater.shape)

    tgt_eq = Tensor(
        [
            [
                5,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -1,
                -1,
                0,
                -1,
                -1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                -1,
                -1,
                1,
                2,
                1,
                0,
                0,
                0,
                1,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                -1,
                -1,
                0,
            ],
            [
                -3,
                0,
                0,
                0,
                0,
                -1,
                -1,
                0,
                1,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
            ],
        ]
    ).reshape(3, 1, 1, 28)
    tgt_shape = ConvAbstractShape(
        tgt_eq, tgt_eq, None, None, c_in=None, n_in=None, padding=None, k=None, stride=None
    )
    # print(tgt_shape.y_greater.shape)

    print(out_shape.y_greater)
    print(tgt_shape.y_greater)

    assert torch.allclose(out_shape.y_greater, tgt_shape.y_greater)


def test_aconv_backsub_conv_tightens_bounds1():
    """ If we increase eps, the new shape should be a superset of the old one.
    """
    C = 1
    C1 = 1
    inputs = torch.zeros(1, 4, 4)
    input_ashape = create_abstract_input_shape(inputs, 1)
    conv_trans1 = AbstractConvolution(
        torch.ones(C1, 1, 4, 4), # kernel
        torch.zeros(C1),  # bias
        2, # stride
        1 # padding
    )
    conv_trans2 = AbstractConvolution(
        torch.ones(C, C1, 4, 4),
        torch.zeros(C),
        2,
        1
    )

    ashape1 = conv_trans1.forward(input_ashape)
    ashape2 = conv_trans2.forward(ashape1)
    ashape2_no_padd = ashape2.zero_out_padding_weights()
    new_ashape = ashape2_no_padd.backsub(ashape1).zero_out_padding_weights()
    recompute_bounds(input_ashape, new_ashape, new_ashape)

    print("new_ashape", new_ashape, sep='\n')

    print("Bad idx lower")
    bad_idx_lower = ashape2.lower > new_ashape.lower
    print(ashape2.lower[bad_idx_lower][:5])
    print("<=")
    print(new_ashape.lower[bad_idx_lower][:5])

    print("Bad idx upper")
    bad_idx_upper = ashape2.upper < new_ashape.upper
    print(ashape2.upper[bad_idx_upper][:5])
    print(">=")
    print(new_ashape.upper[bad_idx_upper][:5])

    assert torch.all(ashape2.lower <= new_ashape.lower)
    assert torch.all(new_ashape.upper <= ashape2.upper)


def test_aconv_backsub_conv_tightens_bounds2():
    """ Backsub gives tighter bounds.
    """
    M = 1
    C = 4
    C1 = 4
    N2 = 28
    inputs = torch.zeros(1, N2, N2)
    input_ashape = create_abstract_input_shape(inputs, M, bounds=(-M, M))
    conv_trans1 = AbstractConvolution(
        M * torch.rand(C1, 1, 4, 4), # kernel
        M * torch.rand(C1),  # bias
        2, # stride
        1 # padding
    )
    conv_trans2 = AbstractConvolution(
        M * torch.rand(C, C1, 4, 4),
        M * torch.rand(C),
        2,
        1
    )

    ashape1 = conv_trans1.forward(input_ashape)
    ashape2 = conv_trans2.forward(ashape1)
    ashape2_no_padd = ashape2.zero_out_padding_weights()
    new_ashape = ashape2_no_padd.backsub(ashape1).zero_out_padding_weights()
    recompute_bounds(input_ashape, new_ashape, new_ashape)

    print("new_ashape", new_ashape, sep='\n')

    print("Bad idx lower")
    bad_idx_lower = ashape2.lower > new_ashape.lower
    print(ashape2.lower[bad_idx_lower][:5])
    print("<=")
    print(new_ashape.lower[bad_idx_lower][:5])

    print("Bad idx upper")
    bad_idx_upper = ashape2.upper < new_ashape.upper
    print(ashape2.upper[bad_idx_upper][:5])
    print(">=")
    print(new_ashape.upper[bad_idx_upper][:5])

    assert torch.all((ashape2.lower <= new_ashape.lower) |
                torch.allclose(ashape2.lower, new_ashape.lower))
    assert torch.all((new_ashape.upper <= ashape2.upper) |
                torch.allclose(new_ashape.upper, ashape2.upper))


def test_aconv_backsub_relu_conv_tightens_bounds3():
    """ Backsub tightenes the bounds.
    """
    M = 10
    C = 32
    C1 = 16
    N2 = 28
    inputs = torch.zeros(1, N2, N2)
    input_ashape = create_abstract_input_shape(inputs, M, bounds=(-M, M))
    conv_trans1 = AbstractConvolution(
        M * torch.rand(C1, 1, 4, 4), # kernel
        M * torch.rand(C1),  # bias
        2, # stride
        1 # padding
    )
    arelu_trans = AbstractReLU('zeros')
    conv_trans2 = AbstractConvolution(
        M * torch.rand(C, C1, 4, 4),
        M * torch.rand(C),
        2,
        1
    )

    conv_shape1 = conv_trans1.forward(input_ashape)
    relu_shape = arelu_trans.forward(conv_shape1)
    conv_shape2 = conv_trans2.forward(relu_shape)
    conv_shape2_no_pad = conv_shape2.zero_out_padding_weights()

    new_ashape = conv_shape2_no_pad\
                .backsub(relu_shape)\
                .zero_out_padding_weights()\
                .backsub(conv_shape1)

    recompute_bounds(input_ashape, new_ashape, new_ashape)

    # print("new_ashape", new_ashape, sep='\n')

    print("Bad idx lower")
    bad_idx_lower = conv_shape2.lower > new_ashape.lower
    print(conv_shape2.lower[bad_idx_lower][:5])
    print("<=")
    print(new_ashape.lower[bad_idx_lower][:5])

    print("Bad idx upper")
    bad_idx_upper = conv_shape2.upper < new_ashape.upper
    print(conv_shape2.upper[bad_idx_upper][:5])
    print(">=")
    print(new_ashape.upper[bad_idx_upper][:5])

    assert torch.all((conv_shape2.lower <= new_ashape.lower) |
                torch.isclose(conv_shape2.lower, new_ashape.lower))
    assert torch.all((new_ashape.upper <= conv_shape2.upper) |
                torch.isclose(new_ashape.upper, conv_shape2.upper))


def test_aconv_backsub_conv_concrete_stays_inside():
    """ The concrete point stays in ashape after backsub.
    """
    # Hyperparams
    M = 1
    C = 32
    C1 = 16
    C2 = 1
    N2 = 28
    K = 4
    S = 2
    P = 1

    # inputs
    inputs = torch.ones(C2, N2, N2)
    input_ashape = create_abstract_input_shape(inputs, 10, bounds=(-1000, 1000))
    inputs = inputs.unsqueeze(0)

    # conv1
    kernel1 = M * torch.rand(C1, C2, K, K)
    bias1 = M * torch.rand(C1)
    conv1 = torch.nn.Conv2d(C2, C1, K, S, P)
    conv1.weight.data = kernel1
    conv1.bias.data = bias1
    conv_trans1 = AbstractConvolution(
        kernel1, # kernel
        bias1,  # bias
        S, # stride
        P # padding
    )
    
    # conv2
    kernel2 = M * torch.rand(C, C1, K, K)
    bias2 = M * torch.rand(C)
    conv2 = torch.nn.Conv2d(C1, C, K, S, P)
    conv2.weight.data = kernel2
    conv2.bias.data = bias2
    conv_trans2 = AbstractConvolution(
        kernel2, # kernel
        bias2,  # bias
        S, # stride
        P # padding
    )

    # Abstract forward
    conv_shape1 = conv_trans1.forward(input_ashape)
    conv_shape2 = conv_trans2.forward(conv_shape1)

    # Concrete forward
    conv1_out = conv1.forward(inputs)
    conv2_out = conv2.forward(conv1_out)
    conv2_out = conv2_out.squeeze(0)

    # Check inlcusion
    print("Bad idx lower")
    bad_idx_lower = conv_shape2.lower > conv2_out
    print("lower", conv_shape2.lower[bad_idx_lower][:5])
    print("<=")
    print("point", conv2_out[bad_idx_lower][:5])

    print("Bad idx upper")
    bad_idx_upper = conv_shape2.upper < conv2_out
    print("upper", conv_shape2.upper[bad_idx_upper][:5])
    print(">=")
    print("point", conv2_out[bad_idx_upper][:5])

    assert torch.all((conv_shape2.lower <= conv2_out) |
                torch.isclose(conv_shape2.lower, conv2_out))
    assert torch.all((conv2_out <= conv_shape2.upper) |
                torch.isclose(conv_shape2.upper, conv2_out))

    # Abstract backward
    conv_shape2_no_pad = conv_shape2.zero_out_padding_weights()
    new_ashape = conv_shape2\
                .backsub(conv_shape1)\
                # .zero_out_padding_weights()
    recompute_bounds(input_ashape, new_ashape, conv_shape2)

    # Check inclusion
    print("Bad idx lower")
    bad_idx_lower = conv_shape2.lower > conv2_out
    print("lower", conv_shape2.lower[bad_idx_lower][:5])
    print("<=")
    print("point", conv2_out[bad_idx_lower][:5])

    print("Bad idx upper")
    bad_idx_upper = conv_shape2.upper < conv2_out
    print("upper", conv_shape2.upper[bad_idx_upper][:5])
    print(">=")
    print("point", conv2_out[bad_idx_upper][:5])

    assert torch.all((conv_shape2.lower <= conv2_out) |
                torch.isclose(conv_shape2.lower, conv2_out))
    assert torch.all((conv2_out <= conv_shape2.upper) |
                torch.isclose(conv_shape2.upper, conv2_out))


def test_aconv_backsub_relu_conv_concrete_stays_inside():
    """ The concrete point stays in ashape after backsub.
    """
    # Hyperparams
    C2 = 1
    N2 = 4
    C1 = 2
    C = 1
    K = 4
    S = 2
    P = 1
    M = 2

    # inputs
    inputs = 0.1 * torch.ones(C2, N2, N2)
    input_ashape = create_abstract_input_shape(inputs, 0.5, bounds=(-10, 10))
    inputs = inputs.unsqueeze(0)
    # print("input_ashape", input_ashape, '\n')

    # conv1
    kernel1 = M * torch.rand(C1, C2, K, K)
    bias1 = M * torch.zeros(C1)
    conv1 = torch.nn.Conv2d(C2, C1, K, S, P)
    conv1.weight.data = kernel1
    conv1.bias.data = bias1
    conv_trans1 = AbstractConvolution(
        kernel1, # kernel
        bias1,  # bias
        S, # stride
        P # padding
    )

    # relu
    relu = torch.nn.ReLU()
    arelu_trans = AbstractReLU('zeros')
    
    # conv2
    kernel2 = M * torch.rand(C, C1, K, K)
    bias2 = M * torch.zeros(C)
    conv2 = torch.nn.Conv2d(C1, C, K, S, P)
    conv2.weight.data = kernel2
    conv2.bias.data = bias2
    conv_trans2 = AbstractConvolution(
        kernel2,
        bias2,
        S,
        P
    )

    # Abstract forward
    conv_shape1 = conv_trans1.forward(input_ashape)
    relu_shape = arelu_trans.forward(conv_shape1)
    conv_shape2 = conv_trans2.forward(relu_shape)

    # print("conv_shape2", conv_shape2, sep='\n')

    # Concrete forward
    conv1_out = conv1.forward(inputs)
    relu_out = relu(conv1_out)
    conv2_out = conv2.forward(relu_out)
    conv2_out = conv2_out.squeeze(0)

    

    # Check inlcusion
    assert torch.all((conv_shape2.lower <= conv2_out) |
                torch.isclose(conv_shape2.lower, conv2_out))
    assert torch.all((conv2_out <= conv_shape2.upper) |
                torch.isclose(conv_shape2.upper, conv2_out))

    # Abstract backward
    conv_shape2_no_pad = conv_shape2.zero_out_padding_weights()
    new_ashape = conv_shape2_no_pad\
                .backsub(relu_shape)\
                .zero_out_padding_weights()\
                .backsub(conv_shape1)
    print("input_ashape", input_ashape, '\n')
    recompute_bounds(input_ashape, new_ashape, conv_shape2)

    # print("conv_shape2", conv_shape2, sep='\n')

    # Check inclusion
    print("Bad idx lower")
    bad_idx_lower = conv_shape2.lower > conv2_out
    print("lower", conv_shape2.lower[bad_idx_lower][:5])
    print("<=")
    print("point", conv2_out[bad_idx_lower][:5])

    print("Bad idx upper")
    bad_idx_upper = conv_shape2.upper < conv2_out
    print("upper", conv_shape2.upper[bad_idx_upper][:5])
    print(">=")
    print("point", conv2_out[bad_idx_upper][:5])

    assert torch.all((conv_shape2.lower <= conv2_out) |
                torch.isclose(conv_shape2.lower, conv2_out))
    assert torch.all((conv2_out <= conv_shape2.upper) |
                torch.isclose(conv_shape2.upper, conv2_out))


def test_linear_backsub_relu():
    cur_shape = LinearAbstractShape(
        torch.tensor([[1., -1, 0.5, -2, 1], [-2., 0.5, -1, 1, -0.5]]),
        torch.tensor([[1., -1, 0.5, -2, 1], [-2., 0.5, -1, 1, -0.5]]),
        torch.tensor([-4, -6]),
        torch.tensor([4.5, -1]),
    )
    prev_shape = ReluAbstractShape(
        torch.tensor([
            [0.],
            [1],
            [0],
            [0]
        ]),
        torch.tensor([
            [0, 0],
            [0, 1],
            [0.75, 0.75],
            [1, 0.5]
        ]),
        torch.tensor([0., 2, 0, 0]),
        torch.tensor([0., 3, 3, 2]),
    ).expand()

    pprev_ashape = LinearAbstractShape(
        None,
        None,
        torch.tensor([-2., 2, -1, -2]),
        torch.tensor([-1., 3, 3, 2]),
    )

    print("pprev_ashape", pprev_ashape, sep='\n', end='\n\n')
    abstract_relu = AbstractReLU('zeros')
    relu_ashape = abstract_relu.forward(pprev_ashape).expand()
    print("relu_ashape", relu_ashape, sep='\n', end='\n\n')
    abstract_linear = AbstractLinear(torch.tensor([[1., -1, 0.5, -2, 1], [-2., 0.5, -1, 1, -0.5]]))
    linear_ashape = abstract_linear.forward(relu_ashape)
    print("linear_ashape", linear_ashape, sep='\n', end='\n\n')
    new_ashape = linear_ashape.backsub(relu_ashape)
    print("new_ashape", new_ashape, sep='\n', end='\n\n')
    recompute_bounds(pprev_ashape, new_ashape, linear_ashape)
    print("pprev_ashape", pprev_ashape, sep='\n', end='\n\n')
    print("recomputed linear_ashape", linear_ashape, sep='\n', end='\n\n')

    new_ashape = cur_shape.backsub(prev_shape)
    assert torch.allclose(
        new_ashape.y_greater, torch.tensor([[-0.5, 0, 0.5, -1.5, 0], [-2.5, 0, -1, 0, -0.25]])
    )
    assert torch.allclose(
        new_ashape.y_less, torch.tensor([[2, 0, 0.5, 0, 0.5], [-1.25, 0, -1, 0.75, 0]])
    )


def test_aconv_backsub_relu():
    curr_eq_greater = torch.ones(1, 2, 2, 9)
    curr_eq_greater *= torch.arange(4).reshape(1, 2, 2, 1)
    curr_eq_greater[0,0,1,5:] -= 2
    curr_eq_less = 2*torch.ones(1, 2, 2, 9)
    curr_eq_less -= (1 + torch.arange(4)).reshape(1, 2, 2, 1)
    curr_conv_shape = ConvAbstractShape(
        curr_eq_greater,
        curr_eq_less,
        torch.tensor([
            [0, 0],
            [2, 3]
        ]), 
        torch.tensor([
            [4, 0],
            [-1, -2]
        ]),
        c_in=2, n_in=2, k=2, padding=1, stride=2
    )

    prev_conv_shape = ConvAbstractShape(
        torch.empty(2, 2, 2, 1),
        torch.empty(2, 2, 2, 1),
        torch.tensor([
            [[-2., 1],
             [0, -2]],

            [[1, 1],
             [-3, -1]]
        ]),
        torch.tensor([
            [[-1., 2],
             [1, 2]],

            [[3, 2],
             [-2, 1]]
        ]),
        c_in=2, n_in=2, k=2, padding=1, stride=2
    )
    
    print("prev_conv_shape", prev_conv_shape, sep='\n', end='\n\n')
    arelu_trans = AbstractReLU('zeros')
    relu_ashape = arelu_trans.forward(prev_conv_shape)
    print("relu_ashape", relu_ashape, sep='\n', end='\n\n')
    print("curr_conv_shape", curr_conv_shape, sep='\n', end='\n\n')
    new_ashape = curr_conv_shape.backsub(relu_ashape)
    print("new_ashape", new_ashape, sep='\n', end='\n\n')
    recompute_bounds(prev_conv_shape, new_ashape, curr_conv_shape)
    print("curr_conv_shape", curr_conv_shape, sep='\n', end='\n\n')

    assert torch.allclose(curr_conv_shape.lower, Tensor([
        [0, 0],
        [2, 3]
    ]))
    assert torch.allclose(curr_conv_shape.upper, Tensor([
        [4, 0],
        [-1, -2]
    ]))


def test_recompute_bounds_conv():
    curr_eq_greater = torch.ones(1, 2, 2, 9)
    curr_eq_greater *= torch.arange(4).reshape(1, 2, 2, 1)
    curr_eq_greater[0,0,1,5:] -= 2
    curr_eq_less = 2*torch.ones(1, 2, 2, 9)
    curr_eq_less -= (1+torch.arange(4)).reshape(1, 2, 2, 1)
    curr_shape = ConvAbstractShape(
        curr_eq_greater,
        curr_eq_less,
        None, None,
        c_in=2, n_in=2, k=2, padding=1, stride=2
    )

    prev_shape = AbstractShape(
        torch.empty(2, 2, 2, 1),
        torch.empty(2, 2, 2, 1),
        torch.tensor([
            [[0., -1],
             [-2, 0]],

            [[0, 1],
             [0, 2]]
        ]),
        torch.tensor([
            [[1., 2],
             [-1, 1]],

            [[2, 1],
             [1, 3]]
        ])
    )

    print("prev_shape", prev_shape, sep='\n', end='\n\n')
    print("curr_shape", curr_shape, sep='\n', end='\n\n')
    recompute_bounds(prev_shape, curr_shape, curr_shape)
    print("curr_shape", curr_shape, sep='\n', end='\n\n')
    
    assert torch.allclose(curr_shape.lower, torch.tensor([
        [0., -1],
        [-2, 9]
    ]))

    assert torch.allclose(curr_shape.upper, torch.tensor([
        [4., 0],
        [1, -6]
    ]))

def main():
    for i in range(100):
        test_aconv_backsub_relu_conv_concrete_stays_inside()
        print(i)

if __name__ == "__main__":
    main()


# def test_ConvAbstractShape_zero_out_padding():
#     curr_eq = torch.ones(2, 2, 2, 33)
#     curr_eq[...,0] = 0
#     curr_shape = ConvAbstractShape(
#         curr_eq, curr_eq, None, None, c_in=2, k=4, n_in=4, padding=1, stride=2
#     )
#     curr_shape.zero_out_padding_weights()

#     print("y_greater")
#     print(curr_shape.y_greater, end='\n\n')
#     print("y_less")
#     print(curr_shape.y_less, end='\n\n')
#     print("y_greater[0,0]")
#     print(curr_shape.y_greater[0,0,0,1:].reshape(2, 10, 10))
#     print("y_greater[0,0] zeroed")

# def test_ConvAbstractShape_zero_out_padding_2():
#     curr_eq = torch.ones(2, 1, 1, 9)
#     curr_eq[...,0] = 0
#     curr_shape = ConvAbstractShape(
#         curr_eq, curr_eq, None, None, c_in=2, k=2, n_in=2, padding=0, stride=2
#     )

#     print("y_greater")
#     print(curr_shape.y_greater, end='\n\n')
#     print("y_less")
#     print(curr_shape.y_less, end='\n\n')
#     curr_shape.zero_out_padding_weights()
#     print("y_greater")
#     print(curr_shape.y_greater, end='\n\n')
#     print("y_less")
#     print(curr_shape.y_less, end='\n\n')
#     _ = 5


# def test_test_ConvAbstractShape_backsub_3():
#     curr_eq = torch.ones(2, 2, 2, 33)
#     curr_eq[...,0] = 0
#     curr_shape = ConvAbstractShape(
#         curr_eq, curr_eq, None, None, c_in=2, k=4, n_in=4, padding=1, stride=2
#     )
#     curr_shape.zero_out_padding_weights()

#     prev_eq = torch.ones(2, 4, 4, 33)
#     prev_eq[...,0] = 0
#     prev_shape = ConvAbstractShape(
#         prev_eq, prev_eq, None, None, c_in=2, n_in=8, k=4, padding=1, stride=2
#     )

#     out_shape = curr_shape.backsub_conv(prev_shape)

#     print("y_greater")
#     print(out_shape.y_greater, end='\n\n')
#     print("y_less")
#     print(out_shape.y_less, end='\n\n')
#     print("y_greater[0,0]")
#     print(out_shape.y_greater[0,0,0,1:].reshape(2, 10, 10))
#     print("y_greater[0,0] zeroed")
#     # out_shape.zero_out_padding_weights()
#     print(out_shape.y_greater[0,0,0,1:].reshape(2, 10, 10))
#     _ = 5

#     anet = AbstractNetwork([])
#     input_shape = AbstractShape(
#         y_greater = torch.ones(2, 8, 8, 1),
#         y_less = -torch.ones(2, 8, 8, 1),
#         upper = torch.ones(2, 8, 8),
#         lower = -torch.ones(2, 8, 8)
#     )
#     new_bounds = anet.recompute_bounds_conv(input_shape, out_shape)
#     print(new_bounds[0], new_bounds[1], sep='\n')

# def test_test_ConvAbstractShape_backsub_4():
#     curr_eq = torch.ones(2, 1, 1, 33)
#     curr_eq[...,0] = 0
#     curr_shape = ConvAbstractShape(
#         curr_eq, curr_eq, None, None, c_in=2, k=4, n_in=4, padding=1, stride=2
#     )
#     curr_shape.zero_out_padding_weights()

#     prev_eq = torch.ones(2, 4, 4, 33)
#     prev_eq[...,0] = 0
#     prev_shape = ConvAbstractShape(
#         prev_eq, prev_eq, None, None, c_in=2, n_in=8, k=4, padding=1, stride=2
#     )

#     out_shape = curr_shape.backsub_conv(prev_shape)

#     print("y_greater")
#     print(out_shape.y_greater, end='\n\n')
#     print("y_less")
#     print(out_shape.y_less, end='\n\n')
#     print("y_greater[0,0]")
#     print(out_shape.y_greater[0,0,0,1:].reshape(2, 10, 10))
#     print("y_greater[0,0] zeroed")
#     # out_shape.zero_out_padding_weights()
#     print(out_shape.y_greater[0,0,0,1:].reshape(2, 10, 10))
#     _ = 5

#     anet = AbstractNetwork([])
#     input_shape = AbstractShape(
#         y_greater = torch.ones(2, 8, 8, 1),
#         y_less = -torch.ones(2, 8, 8, 1),
#         upper = torch.ones(2, 8, 8),
#         lower = -torch.ones(2, 8, 8)
#     )
#     new_bounds = anet.recompute_bounds_conv(input_shape, out_shape)
#     print(new_bounds[0], new_bounds[1], sep='\n')