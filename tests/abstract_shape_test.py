import torch
import sys
sys.path.append("./code")

from abstract_shape import ReluAbstractShape

def test_expand_abstract_shape():
    abstract_shape = ReluAbstractShape(
        torch.tensor([[1], [2]]),
        torch.tensor([[3, 4], [5, 6]]),
        torch.tensor([7, 8]),
        torch.tensor([9, 10])
    )

    out = abstract_shape.expand()
    
    assert torch.allclose(out.y_greater, torch.tensor([[0, 1, 0], [0, 0, 2]]))
    assert torch.allclose(out.y_less, torch.tensor([[3, 4, 0], [5, 0, 6]]))
    assert torch.allclose(out.lower, torch.tensor([7, 8]))
    assert torch.allclose(out.upper, torch.tensor([9, 10]))

# def test_alinear_backsub():
#     cur_shape = LinearAbstractShape(
#         torch.ones(2, 4),
#         -torch.ones(2, 4),
#         torch.ones(2),
#         -torch.ones(2),
#     )
#     prev_shape = ReluAbstractShape(
#         2*torch.ones(3, 5),
#         torch.zeros(3, 5),
#         torch.ones(3),
#         torch.ones(3),
#     )

#     new_shape1 = cur_shape.backsub(prev_shape)
#     new_shape2 = cur_shape.backsub2(prev_shape)

#     assert new_shape.shape == (2, 5)
#     assert torch.allclose(new_shape.y_greater, torch.tensor(
#         [[]]
#     ))
#     assert torch.allclose(new_shape.y_less, new_shape2.y_less)