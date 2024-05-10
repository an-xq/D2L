#softmax回归的简洁实现
import torch
from torch import nn
from d2l import torch as d2l
from IPython import display #图像显示

#1.读取数据集
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

#2.定义模型，初始化模型参数
net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))#一个展平层，一个全连接层

def init_weights(m):
    if type(m)==nn.Linear:#对全连接层初始化
        nn.init.normal_(m.weight,std=0.01)#对m.weight初始化，均值默认为0，方差为0.01
net.apply(init_weights)#接受一个函数作为参数，并将这个函数应用到模型的每个参数张量上

#3.损失函数
loss=nn.CrossEntropyLoss(reduction='none')#参数 reduction='none' 指定了损失的计算方式。在这里，'none' 表示不进行损失的降维（reduction），即保留每个样本的损失值，不对损失进行平均或求和。

#4.优化算法
trainer=torch.optim.SGD(net.parameters(),lr=0.1)

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
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())#加入本batch的总交叉熵，预测正确的个数，batch总数
    return metric[0]/metric[2],metric[1]/metric[2]#返回平均交叉熵和预测成功的比例

def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics=train_epoch_ch3(net,train_iter,loss,updater)
        test_acc=evaluate_accuracy(net,test_iter)
        animator.add(epoch+1,train_metrics+(test_acc,))
    train_loss,train_acc=train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)