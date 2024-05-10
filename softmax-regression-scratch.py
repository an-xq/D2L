#softmax回归的从零开始实现
import torch
import torchvision
from torch.utils import data #数据处理
from torchvision import transforms #图像处理
from IPython import display #图像显示
from d2l import torch as d2l

#d2l.use_svg_display() jupyter里用了

#1.读取数据集

trans=transforms.ToTensor() #通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式,并除以255使得所有像素的数值均在0～1之间
mnist_train=torchvision.datasets.FashionMNIST(root="./data",train=True,transform=trans,download=True)
mnist_test=torchvision.datasets.FashionMNIST(root="./data",train=False,transform=trans,download=True)

# print(len(mnist_train),len(mnist_test))
# print(mnist_train[0][0].shape)#第一个[0]是训练集的图片，[1]是标注，第二个[0]指代第一个样本，所以这句话代表第一个样本图片的形状

def get_fashion_mnist_labels(labels):
    text_labels=['t-shirt', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return[text_labels[int(i)] for i in labels]

# print(get_fashion_mnist_labels([1,2,3]))

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #绘制图像列表
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):# 图片张量
            ax.imshow(img.numpy())
        else:# PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    d2l.plt.show()
    return axes

# X,y=next(iter(data.DataLoader(mnist_train,batch_size=18)))
# show_images(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))#画出来2行，每行9个

def get_dataloader_workers():
    return 0 #进程数，也可以在下边直接传数字，但是linux可以开多个,试了一下，开1个是16秒，开4个是10秒，win10及以下好像不能多开，只能传0

# batch_size=256
# train_iter=data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers())
# timer=d2l.Timer()#输出取数时间
# for X,y in train_iter:
#     continue
# print(f'{timer.stop():.2f}sec')


def load_data_fashion_mnist(batch_size, resize=None):#下载Fashion-MNIST数据集，然后将其加载到内存中
    trans = [transforms.ToTensor()]#将ToTensor()转换操作放在了一个列表中
    if resize:#Resize()从小改大一般用插值法
        trans.insert(0, transforms.Resize(resize))#在转换操作序列 trans 的开头插入一个transforms.Resize(resize) 操作,使得转换操作列表变为 [transforms.Resize(resize), transforms.ToTensor()]
    trans = transforms.Compose(trans)#将多个转换操作组合成一个序列
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

batch_size=256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# for X, y in train_iter:
#     print(X.shape, X.dtype, y.shape, y.dtype)
#     break


#2.初始化模型参数,实际上X-256*784，W-784*10，b-10，y-256*10
num_inputs=784
num_outputs=10

W=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b=torch.zeros(num_outputs,requires_grad=True)

#3.softmax操作，即对结果归一化
def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True) #对第1个维度求和，即对y的每行求和
    return X_exp/partition

#4.模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W)+b)

#5.损失函数loss
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])
#y_hat[range(len(y_hat)),y]取出每个样本的10个概率中的标注的那个正确种类的预测概率
#取log，然后求负，就是我们这里的交叉熵

#6.分类精度，用来获取预测的正确情况
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

#7.训练

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

#8.优化算法
lr=0.1
def updater(batch_size):
    return d2l.sgd([W,b],lr,batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

