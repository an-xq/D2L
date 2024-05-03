#线性回归的从零开始实现
#第一步，import一些你需要的包
import os #与操作系统交互的库
import matplotlib.pyplot as plt#matplotlib 库的一个子模块,用来画图
import random
import torch
from d2l import torch as d2l#from的意思是从d2l包中仅import torch模块，并以d2l为别名

#第二步，声明生成数据集的函数,w和b即为待求参数，这里我们自己定
def synthetic_data(w,b,num_examples):#用自己定的w,b，还有预期的训练样本数目num_examples来生成训练数据，y=Xw+b+噪声，synthetic是合成的意思
    X=torch.normal(0,1,(num_examples,len(w)))#torch.normal(mean, std, size) mean是正态分布的均值，std是标准差，后边的size是shape，生成1000个符合正态分布的随机样本，每个样本有两个元素，即1000*2的矩阵
    y=torch.matmul(X,w)+b #matmul是矩阵乘法，通过自己定的w和b,计算训练样本的y
    y+=torch.normal(0,0.01,y.shape)#人为添加噪音
    return X,y.reshape((-1,1))

#第三步，声明真实的w和b,并利用第二步的函数生成训练集
true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)

#中途检查
print('features:',features[0],'\nlabel:',labels[0])#检查生成的第一个样本
d2l.set_figsize()#设置图形尺寸为默认
d2l.plt.scatter(features[:,(1)].detach().numpy(),labels.detach().numpy(),1);#plt是绘图函数，scatter指散点图，features[:,(1)].detach().numpy()是第一个参数即横坐标，取features的所有行，第1列，detach是指将其分离出来，然后转换为numpy，才能绘制散点图。labels.detach().numpy()是第二个参数即纵坐标，1是散点大小
plt.show()#展示图像的

#第四步，读取生成好的训练集，给搞成方便学习的样子
def data_iter(batch_size,features,labels):#声明生成数据迭代器函数，传入小批量大小batch_size、训练集features,labels
    num_examples=len(features)#训练集样本数量
    indices=list(range(num_examples))#indices指数的意思，生成一个从0到1000的列表
    random.shuffle(indices)#把indices列表打乱
    for i in range(0,num_examples,batch_size):#range用于生成一个起始于 0、步长为 batch_size、不超过 num_examples 的整数序列,即i按照步长跳着取
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])#创建张量，当i=0时从indices这个打乱的列表里取前0-9个数作为一个张量，最后一组取不够10个的话，能取到哪算哪
        yield features[batch_indices],labels[batch_indices]#高级的return，一次返回一组，后边配合Next来实现多次返回

#设置参数
batch_size=10
#中途检查
for X,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break

#第五步，初始化模型参数
w=torch.normal(0,0.01,size=(2,1),requires_grad=True)#requires_grad=True表示这些生成的随机数需要计算梯度
b=torch.zeros(1,requires_grad=True)
#因为下边要对w和b梯度下降，所以这里先声明一个，然后赋个随机值，并且将梯度计算置为true，以便接受反向传递回的梯度

#第六步，定义模型，即我们要求的w和b对应了什么样的模型
def linreg(X,w,b):
    return torch.matmul(X,w)+b#matmul矩阵乘法

#第七步，定义损失函数
def squared_loss(y_hat,y):#传入的是两个向量，形状为batch_size长度的列向量
    return(y_hat-y.reshape(y_hat.shape)) ** 2/2#均方损失

#第八步，定义优化算法
def sgd(params,lr,batch_size):#定义随机梯度下降算法，params是w加b，lr是学习率
    with torch.no_grad():#with开启一个上下文，即在下文的更新过程中不必计算梯度，提高效率
        for param in params:#第一次取出w，是个(2,1)的向量，第二次取出b，是个标量，由于b是一个标量，param.grad也会是一个标量，减去学习率乘以梯度后，更新操作会直接应用于b。而w作为一个向量，其梯度更新会沿着其维度进行。
            param-=lr*param.grad/batch_size#梯度下降方向更新
            param.grad.zero_()#梯度需手动清零，不然会自动累加

#第九步，训练
lr=0.03#手动设置学习率
num_epochs=3#迭代周期数
net=linreg#第六步线性模型作为网络层
loss=squared_loss#第七步损失函数

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):#将第三步的数据集，传入第四步进行调整，然后取出一组X，y
        l=loss(net(X,w,b),y)#net(X,w,b)计算出当前参数下的y_hat,再将y_hat和y传入loss,此时l也应该是一个长度为batch_size的列向量
        l.sum().backward()#对l求和，然后backward反向传递梯度
        sgd([w,b],lr,batch_size)#调用梯度下降优化
    with torch.no_grad():#打印每一轮损失
        train_l=loss(net(features,w,b),labels)
        print(f'epoch{epoch+1},loss{float(train_l.mean()):f}')#f前缀表示这是一个f-string，允许你在花括号{}中直接包含Python表达式。:f: 这是格式化指令，用于指定浮点数的格式化方式,而非科学计数法之类的。

#验证
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差：{true_b-b}')



print('Hello World')
