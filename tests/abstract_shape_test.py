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
)
from abstract_networks import AbstractNetwork


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


def main():
    test_ConvAbstractShape_zero_out_padding_2()

if __name__ == "__main__":
    main()