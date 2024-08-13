import os
import torch
from d2l import torch as d2l
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import math
from torch import nn


#数据生成
max_degree=20
n_train=100
n_test=100
batch_size=10

true_w=np.zeros(max_degree)
true_w[0:4]=np.array([5,1.2,-3.4/math.factorial(2),5.6/math.factorial(3)])
#训练数据
train_features=torch.normal(0,1,(n_train,1))
temp_train_labels=np.power(train_features,np.arange(max_degree))
train_labels=temp_train_labels @ torch.from_numpy(true_w)
train_labels=train_labels.float()
train_labels+=torch.normal(0,0.01,train_labels.shape)
#验证数据
test_features=torch.normal(0,1,(n_test,1))
temp_test_labels=np.power(test_features,np.arange(max_degree))
test_labels=temp_test_labels @ torch.from_numpy(true_w)
test_labels=test_labels.float()
test_labels+=torch.normal(0,0.01,test_labels.shape)

#数据迭代器
def data_iter(data_arrs,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_arrs)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

train_iter=data_iter([train_features,train_labels],batch_size)
test_iter=data_iter([test_features,test_labels],batch_size)

#损失
loss = nn.MSELoss()

#优化算法
def sgd(param,lr,batch_size):
    with torch.no_grad():
        # print(param.grad)
        param-=lr*param.grad/batch_size
        param.grad.zero_()

#模型
def net(X,w,num_w):
    temp=np.power(X,np.arange(num_w))
    temp=temp.float()
    fact_w=w[0:num_w]
    # fact_w=fact_w.float()
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
epochs=50
num_w=4
lr=0.1
w=torch.normal(0,0.01,size=(max_degree,),requires_grad=True)
for epoch in range(epochs):
    for X,y in train_iter:
        y_hat=net(X,w,num_w)
        l=loss(y_hat,y)
        l.mean().backward()
        sgd(w,lr,batch_size)
    train_loss=evaluate_loss(net,train_iter,w,num_w,loss)
    test_loss=evaluate_loss(net,test_iter,w,num_w,loss)
    print("epoch:",epoch)
    print("train_loss:",train_loss)
    print("test_loss:",test_loss)
    









