import os
import torch
from d2l import torch as d2l
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import math
from torch import nn
from IPython import display #图像显示


#数据生成
max_degree=20
n_train=100
n_test=100
batch_size=10

true_w=np.zeros(max_degree)
true_w[0:4]=np.array([5,1.2,-3.4,5.6])
print(" ".join(f"{value:.6f}" for value in true_w))
#训练数据
train_features=torch.normal(0,1,(n_train,1))
temp_train_labels=np.power(train_features,np.arange(max_degree))
for i in range(max_degree):
    temp_train_labels[:,i]/=math.gamma(i+1)
train_labels=temp_train_labels @ torch.from_numpy(true_w)
train_labels=train_labels.float()
train_labels+=torch.normal(0,0.01,train_labels.shape)

#验证数据
test_features=torch.normal(0,1,(n_test,1))
temp_test_labels=np.power(test_features,np.arange(max_degree))
for i in range(max_degree):
    temp_test_labels[:,i]/=math.gamma(i+1)
test_labels=temp_test_labels @ torch.from_numpy(true_w)
test_labels=test_labels.float()
test_labels+=torch.normal(0,0.01,test_labels.shape)

#数据迭代器
def data_iter(data_arrs,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_arrs)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

train_iter=data_iter([train_features,train_labels],batch_size)
test_iter=data_iter([test_features,test_labels],batch_size,is_train=False)

#损失
loss = nn.MSELoss()

#优化算法
def sgd(param,lr,batch_size):
    with torch.no_grad():
        param-=lr*param.grad/batch_size
        param.grad.zero_()

#模型
def net(X,w,num_w):
    temp=np.power(X,np.arange(num_w))
    for i in range(num_w):
        temp[:,i]/=math.gamma(i+1)
    temp=temp.float()
    fact_w=w[0:num_w]
    return temp @ fact_w


#工具
def evaluate_loss(net,data_iter,w,num_w,loss):
    metric=d2l.Accumulator(2)
    for X,y in data_iter:
        y_hat=net(X,w,num_w)
        l=loss(y_hat,y)
        metric.add(l.sum(),d2l.size(l))
    return metric[0]/metric[1]

#训练
epochs=1500
num_w=20
lr=0.01
# w = torch.nn.Linear(num_w,1).weight[0,:]
# print(w)
# w=torch.normal(0,0.01,size=(num_w,),requires_grad=True)
w = torch.tensor([ 0.1443,  0.0954,  0.0023,  0.1622, -0.0402, -0.0688,  0.0541,  0.1384,
        -0.0383,  0.1708, -0.0298,  0.0262, -0.0790,  0.0518, -0.1726,  0.1879,
        -0.1540, -0.0822,  0.0842, -0.1048],requires_grad=True)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
for epoch in range(epochs):
    for X,y in train_iter:
        y_hat=net(X,w,num_w)
        l=loss(y_hat,y)
        l.backward()
        sgd(w,lr,batch_size)
    # print(w)
    train_loss=evaluate_loss(net,train_iter,w,num_w,loss)
    test_loss=evaluate_loss(net,test_iter,w,num_w,loss)
    print("epoch:",epoch)
    print("train_loss:",train_loss)
    print("test_loss:",test_loss)
    if epoch == 0 or (epoch + 1) % 20 == 0:
        animator.add(epoch + 1, (train_loss,test_loss))

print(" ".join(f"{value:.6f}" for value in w))
    









