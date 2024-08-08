import torch

# x = torch.arange(4.0)
# print("x:",x)

# x.requires_grad_(True)
# print("x.grad:",x.grad)

# y = 2 * torch.dot(x,x)
# print("y:",y)

# y.backward()
# print("x.grad:",x.grad)
# print(x.grad == 4*x)

# x.grad.zero_()
# y=x.sum()
# y.backward()
# print("x.grad:",x.grad)

# x.grad.zero_()
# y = x*x
# print("y:",y)
# y.sum().backward()
# print("x.grad:",x.grad)

# x.grad.zero_()
# y = x * x
# u = y.detach()#把u当成一个常数而不是一个关于y的函数
# z = u * x
# print("u:",u)
# z.sum().backward()
# print("x.grad:",x.grad)
# print(x.grad ==u)
# x.grad.zero_()
# y.sum().backward()
# print("x.grad:",x.grad)
# print(x.grad==2*x)


def f(a):
    b = a*2
    while b.norm()<1000:
        b = b * 2
    if b.sum()>0:
        c=b
    else:
        c=100*b
    return c
a = torch.randn(size=(),requires_grad=True)#a是一个随机标量，计算梯度
d = f(a)
d.backward()

print(a.grad)
print(a.grad==d/a)
