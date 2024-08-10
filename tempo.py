import torch
import torchvision
from torch.utils import data

batch_size=256

trans = torchvision.transforms.ToTensor()
mnist_train=torchvision.datasets.FashionMNIST(root="./data",train=True,transform=trans,download=True)
mnist_test=torchvision.datasets.FashionMNIST(root="./data",train=False,transform=trans,download=True)
classes = mnist_train.classes
train_iter = data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True)
test_iter = data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False)

#参数初始化
num_inputs=28*28
num_outputs=10

w=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b=torch.zeros(num_outputs,requires_grad=True)

def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)
    # partition=X_exp.sum()
    return X_exp/partition

#模型
def net(X):
    return softmax(torch.matmul(X.reshape(-1,w.shape[0]),w)+b)


class Accumulator:
    def __init__(self,n):
        self.data=[0.0]*n
        print(self.data)

    def add(self, *args):#星号用来打包传递
        self.data = [a + float(b) for a, b in zip(self.data, args)]#self.data和args两两相加
        # print("a:",a,"\tb:",b)
    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=y_hat.argmax(axis=1)
    cmp=y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())



def evaluate_accuracy(net,data_iter):#计算在指定数据集上模型的精度
    metric=Accumulator(2)#这两个分别为正确预测数、预测总数
    for X,y in data_iter:#不断把迭代器里每一组样本的正确数和总数加进metric
        # print("accuracy(net(X), y):",accuracy(net(X), y),"y.numel():",y.numel())
        metric.add(accuracy(net(X), y),y.numel())#例如正确数是3，y.numel()元素个数是10，args就是(3,10)，打包了
    return metric[0]/metric[1]#返回精度，即正确率

print(evaluate_accuracy(net,train_iter))
