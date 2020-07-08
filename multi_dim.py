import matplotlib.pyplot as plt
import torch
from pdb import set_trace as db


class Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, x3):
        ctx.save_for_backward(x1, x2, x3)
        y1 = x1 + x2
        y2 = x2 + x3
        y3 = x3 + x1
        return y1, y2, y3

    @staticmethod
    def backward(ctx, dy1, dy2, dy3):
        x1, x2, x3 = ctx.saved_tensors
        print("dy: ",dy1, dy2, dy3)
        dx1 = dy1*1 + dy2*0 + dy3*1
        dx2 = dy1*1 + dy2*1 + dy3*0
        dx3 = dy1*0 + dy2*1 + dy3*1

        return dx1, dx2, dx3


if __name__=="__main__":
    x1 = torch.Tensor([0]).requires_grad_()
    x2 = torch.Tensor([1]).requires_grad_()
    x3 = torch.Tensor([2]).requires_grad_()

    my_func = Func.apply

    y1, y2, y3 = my_func(x1, x2, x3)
    L = y1 * y2 * y3
    print("L: ",L)
    L.backward()
    print("x1 grad: ", x1.grad)
    print("x2 grad: ", x2.grad)
    print("x3 grad: ", x3.grad)
