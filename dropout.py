#多层感知机的从零开始实现

import torch
from torch import nn
from d2l import torch as d2l
from IPython import display #图像显示

#定义丢弃函数
def dropout(X,p):
    assert 0<=p<=1
    if p==1:
        return X
    if p==0:
        return torch.zeros_like(X)
    temp=torch.randn(X.shape)
    m=(temp>p).float()
    return m*X /(1.0-p)

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

#1.初始化模型参数
num_inputs,num_outputs,num_hiddens=784,10,256
W1=nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True)*0.01)
b1=nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
W2=nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True)*0.01)
b2=nn.Parameter(torch.zeros(num_outputs,requires_grad=True))
params=[W1,b1,W2,b2]#用列表来管理参数
# print(W1,b1,W2,b2)
#randn均值为0、标准差为1的标准正态分布,*0.01将标准正态分布的随机值缩放到均值为0、标准差为0.01的范围内，即将每个值乘以0.01
#Parameter参数的意思，大概就是pytorch的优化器会判断这个tensor是不是parameters，如果是就会每次迭代都更新这个参数,比如说如果只是torch的tensor，在pytorch写好的优化器里就不会更新
#一般来说，对于模型参数，建议使用nn.Parameter来明确地表示这是一个模型的可学习参数；而对于其他张量，可以直接使用requires_grad=True来指示PyTorch需要计算其梯度。

#2.激活函数
def relu(X):
    a=torch.zeros_like(X)
    return torch.max(X,a)

#3.模型
# def net(X):
#     X=X.reshape((-1,num_inputs))#所以X的形状应该是256*784
#     H=relu(X@W1+b1)#@是矩阵乘法
#     return (H@W2+b2)
dropout1, dropout2 = 0.2, 0.5
class Net(nn.Module):
    def __init__(self,num_inputs,num_outputs,num_hiddens1,num_hiddens2,istraining=True):
        super(Net,self).__init__()
        self.num_inputs=num_inputs
        self.num_outputs=num_outputs
        self.num_hiddens1=num_hiddens1
        self.num_hiddens2=num_hiddens2
        self.istraining=istraining
        self.lin1=nn.Linear(self.num_inputs,self.num_hiddens1)
        self.lin2=nn.Linear(self.num_hiddens1,self.num_hiddens2)
        self.lin3=nn.Linear(self.num_hiddens2,self.num_outputs)
        self.relu=nn.ReLU()
    def forward(self,X):
        H1=self.relu(self.lin1(X.reshape(-1,self.num_inputs)))
        if self.istraining==True:
            H1=dropout(H1,dropout1)
            print("istraining==True")
        else:
            print()
        H2=self.relu(self.lin2(H1))
        if self.istraining==True:
            H2=dropout(H2,dropout2)
            print("istraining==True")
        else:
            print("istraining==False")
        out=self.lin3(H2)
        return out

#4.损失函数
loss=nn.CrossEntropyLoss(reduction='none')#交叉熵，不改变形状

####d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
#5.分类精度，用来获取预测的正确情况
def accuracy(y_hat,y):#计算预测正确的数量
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=y_hat.argmax(axis=1)#y_hat变为自己每一行最大值所在的列号，即每个样本的预测类型号
    cmp=y_hat.type(y.dtype)==y #每个样本预测的类型和标注的类型，相等置为1，不相等置为0
    return float(cmp.type(y.dtype).sum()) #返回总正确个数

class Accumulator:
    def __init__(self,n):
        self.data=[0.0]*n

    def add(self, *args):#星号用来打包传递
        self.data = [a + float(b) for a, b in zip(self.data, args)]#self.data和args两两相加

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net,data_iter):#计算在指定数据集上模型的精度
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric=Accumulator(2)#这两个分别为正确预测数、预测总数
    with torch.no_grad():
        for X,y in data_iter:#不断把迭代器里每一组样本的正确数和总数加进metric
            metric.add(accuracy(net(X), y),y.numel())#例如正确数是3，y.numel()元素个数是10，args就是(3,10)，打包了
    return metric[0]/metric[1]#返回精度，即正确率

#6.训练
class Animator:  #绘图类
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        d2l.plt.pause(0.1)
        display.display(self.fig)
        display.clear_output(wait=True)

def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric=Accumulator(3)
    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
            # print("111111111")
        else:
            l.sum().backward()
            updater(X.shape[0])
            # print("123456789")
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())#加入本batch的总交叉熵，预测正确的个数，batch总数
    return metric[0]/metric[2],metric[1]/metric[2]#返回平均交叉熵和预测成功的比例

def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        net.istraining=True
        train_metrics=train_epoch_ch3(net,train_iter,loss,updater)
        net.istraining=False
        test_acc=evaluate_accuracy(net,test_iter)
        animator.add(epoch+1,train_metrics+(test_acc,))
    train_loss,train_acc=train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

#7.训练
num_epochs,lr=10,0.5
net=Net(num_inputs=784,num_outputs=10,num_hiddens1=256,num_hiddens2=256)
updater=torch.optim.SGD(net.parameters(),lr=lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)


