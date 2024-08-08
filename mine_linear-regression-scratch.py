import os
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
import random

#生成数据
def data_gen(w,b,data_num):
    X=torch.normal(0,1,[data_num,len(w)])
    y=torch.matmul(X,w)+b
    # print(y.shape)
    y+=torch.normal(0,0.01,y.shape)
    return X,y



#数据预处理
def data_iter(features,labels,data_num,batch_size):
    index=list(range(data_num))
    random.shuffle(index)
    # print(index)
    for i in range(0,data_num,batch_size):
        batch_index=torch.tensor(index[i:i+batch_size])
        yield features[batch_index],labels[batch_index]


#模型定义
def model(X,w,b):
    y = torch.matmul(X,w)+b
    return y

#损失函数定义
def loss(y,label):
    # print("loss y shape:",y.size())
    # print("loss label shape:",label.size())
    l=(y-label.reshape(y.shape))**2/2
    # print("loss l shape:",l.size())
    return l

#优化算法定义
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batch_size
            param.grad.zero_()

#训练过程
if __name__=='__main__':
    w_true=torch.tensor([1,-2.0])
    b_true=3.0
    data_num=1000
    features,labels=data_gen(w_true,b_true,data_num)
    print("data is ready")
    
    # d2l.set_figsize()
    # d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
    # plt.show()
    
    #参数初始化
    w=torch.normal(0,0.1,size=[2,1],requires_grad=True)
    b=torch.normal(0,0.1,size=[1],requires_grad=True)
    # b=torch.zeros(1,requires_grad=True)
    print("w,b is initial")

    batch_size=10
    lr=0.03
    epochs=3
    for epoch in range(epochs):
        for X,y in data_iter(features,labels,data_num,batch_size):
            # print("X:",X)
            # print("y:",y)
            # print("w",w)
            l=loss(model(X,w,b),y)
            # print("l:",l.size())
            l.sum().backward()
            sgd([w,b],lr,batch_size)
        with torch.no_grad():
            epoch_l=loss(model(features,w,b),labels)
            # print("第",epoch,"轮loss:",float(epoch_l.mean()))
            print("第{}轮loss:{:.6f}".format(epoch,epoch_l.mean().item()))
    
    # print((w-w_true.reshape(w.size())).size())
    # print((b-b_true).size())
    print("w的误差:{:.6f}".format((w-w_true.reshape(w.size())).mean().item()))
    print("b的误差:{:.6f}".format((b-b_true).item()))
