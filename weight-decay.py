import os
import torch
from d2l import torch as d2l
from torch import nn

#数据生成
num_inputs=100
n_train=20
n_test=100
batch_size=5

true_w=torch.ones((num_inputs,1))*0.01
true_b=0.05

train_data=d2l.synthetic_data(true_w,true_b,n_train)
test_data=d2l.synthetic_data(true_w,true_b,n_test)
train_iter=d2l.load_array(train_data,batch_size,is_train=True)
test_iter=d2l.load_array(test_data,batch_size,is_train=False)

#损失
loss = nn.MSELoss()

#模型
def model(X,w,b):
    y = torch.matmul(X,w)+b
    return y

#定义L2范数惩罚
def L2_penalty(w):
    return torch.sum(w**2)/2

def evaluate_loss(model,data_iter,w,b,loss,lambd):
    metric=d2l.Accumulator(2)
    for X,y in data_iter:
        y_hat=model(X,w,b)
        l=loss(y_hat,y)+lambd*L2_penalty(w)
        metric.add(l.sum(),d2l.size(l))
    return metric[0]/metric[1]

#训练
def train(lambd):
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    epochs=100
    lr=0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, epochs], legend=['train', 'test'])

    trainer = torch.optim.SGD((w,b),lr)

    for epoch in range(epochs):
        for X,y in train_iter:
            l=loss(model(X,w,b),y)+lambd*L2_penalty(w)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (evaluate_loss(model,train_iter,w,b,  loss,lambd),
                                     evaluate_loss(model, test_iter, w,b,loss,lambd)))
    print('w的L2范数是', torch.norm(w).item())

def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs,1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    epochs=100
    lr=0.003
    trainer = torch.optim.SGD([{"params":net[0].weight,'weight_decay': wd},{"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, epochs], legend=['train', 'test'])

    for epoch in range(epochs):
        for X,y in train_iter:
            trainer.zero_grad()
            l=loss(net(X),y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是', net[0].weight.norm().item())



# train(lambd=20)
train_concise(40)
