import os
import torch
from d2l import torch as d2l
from torch.utils import data


#数据加载
w_true = torch.tensor([1.0,2.0])
b_true = 3.0
data_num = 1000
features,labels = d2l.synthetic_data(w_true,b_true,data_num)

#数据迭代器生成
def data_loader(data_arrs,batch_size,is_train=True):
    dataset = data.TensorDataset(*data_arrs)#把features和labels分两个传入，声明一个TensorDataset对象
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batchsize = 10
dataloader=data_loader([features,labels],batchsize)
print("dataloader is ready")

#模型定义
from torch import nn
model = nn.Sequential(nn.Linear(2,1))
model[0].weight.data.normal_(0,0.01)
model[0].bias.data.fill_(0)
print("model is ready")

#损失函数定义
loss = nn.MSELoss()
print("lossfunction is ready")

#优化函数定义
lr=0.02
trainer = torch.optim.SGD(model[0].parameters(),lr)
print("optimizer is ready")
# for name,p in model.named_parameters():
#     print(name)

#训练
if __name__== '__main__':
    epochs = 4
    for epoch in range(epochs):
        for X,y in dataloader:
            l=loss(model(X),y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l=loss(model(features),labels)
        print("epoch:{},l={:.6f}".format(epoch,l))
    w=model[0].weight.data
    print(w.size())
    print(w-w_true.reshape(w.size()).mean())
    b=model[0].bias.data
    print(b.size())
    print("w偏差:{:.6f}".format((w.reshape(w_true.size())-w_true).mean().item()))
    print("b偏差:{:.6f}".format(b.item()-b_true))
