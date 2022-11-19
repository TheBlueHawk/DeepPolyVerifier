import torch
import sys
sys.path.append("./code")

from abstract_shape import ReluAbstractShape, LinearAbstractShape

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

def test_alinear_backsub():
    cur_shape = LinearAbstractShape(
        torch.tensor([[1., -1, 1, -1], 
                     [-1., 1, -1, 1]]),
        torch.tensor([[-1., 1, -1, 1], 
                     [1., -1, 1, -1]]),
        torch.ones(2),
        -torch.ones(2),
    )
    prev_shape = ReluAbstractShape(
        2*torch.ones(3, 5, dtype=torch.float32),
        torch.stack([torch.zeros(5, dtype=torch.float32), torch.ones(5), -torch.ones(5)]),
        torch.ones(3),
        torch.ones(3),
    )

    new_shape = cur_shape.backsub(prev_shape)

    assert torch.allclose(new_shape.y_greater, torch.tensor(
        [[4., 3, 3, 3, 3],
         [2., 3, 3, 3, 3]]
    ))
    assert torch.allclose(new_shape.y_less, torch.tensor(
        [[-4., -3, -3, -3, -3],
         [-2., -3, -3, -3, -3]]
    ))