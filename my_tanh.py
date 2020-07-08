import matplotlib.pyplot as plt
import torch
from pdb import set_trace as db

def d_tanh(x):
    return 1 / (x.cosh() ** 2)


class custom_tanh( torch.autograd.Function ):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward( x )
        h = x / 4.0
        y = 4 * h.tanh()
        return y

    @staticmethod
    def backward(ctx, dL_dy):  # dL_dy = dL/dy
        x, = ctx.saved_tensors
        h = x / 4.0
        dy_dx = d_tanh( h )
        dL_dx = dL_dy * dy_dx

        return dL_dx

def draw_graph(f1, f2):
    x_lin = torch.linspace(-10, 10, 10000)
    y1 = f1(x_lin).numpy()
    y2 = f2(x_lin).numpy()
    plt.plot(x_lin, y1, label='tanh')
    plt.plot(x_lin, y2, label='4tanh(x/4)')
    plt.grid(True)
    plt.legend()
    plt.savefig("./graph_image.png")
    # plt.show()


if __name__=="__main__":
    # 入力
    X = torch.Tensor([0,4]).requires_grad_()
    # 関数化
    my_func = custom_tanh.apply
    # 描画
    # draw_graph(torch.tanh, my_func)
    # 計算を実行
    y = my_func(X)
    L = torch.sum(y*y)
    L.backward()
    print(X.grad)

