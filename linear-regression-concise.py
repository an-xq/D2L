#线性回归的简洁实现
#第一步，import一些你需要的包
import numpy as np#NumPy是一个开源的Python科学计算库，它广泛应用于数据科学、机器学习、科学计算和数据分析等领域。NumPy的主要特点是提供了一个强大的多维数组对象ndarray以及用于操作这些数组的函数集。
import torch
from torch.utils import data#torch.utils.data模块提供了用于数据加载和处理的工具，它允许用户更高效地加载和访问数据集，特别是在训练机器学习模型时。
from d2l import torch as d2l#from的意思是从d2l包中仅import torch模块，并以d2l为别名

#第二步，生成数据集
true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=d2l.synthetic_data(true_w,true_b,1000)#生成数据集的函数李沐在d2l里写好了

#第三步，读取数据集
def load_array(data_arrays,batch_size,is_train=True):#构造一个PyTorch数据迭代器,布尔值is_train表示是否希望数据迭代器对象在每个迭代周期内打乱数据
    dataset=data.TensorDataset(*data_arrays)#data_arrays是一个包含多个元素的列表或元组，使用*data_arrays会将列表或元组中的每个元素作为单独的参数传递给函数。这样做的好处是可以方便地将一个包含多个元素的集合传递给一个接受多个参数的函数，而不需要手动地逐个指定每个参数。实际上TensorDataset这个函数的参数形式就是*tensors，tensors是一个或多个张量，这些张量在数据集中被看作是一组样本，每个样本由这些张量中对应位置的元素组成，最终拆成1000个样本，存在dataset里
    return data.DataLoader(dataset, batch_size,shuffle=is_train)#用上文的TensorDataset对象，进一步生成DataLoader数据迭代器，将输入的数据数组加载到 DataLoader 中，并返回一个可以用于迭代的 DataLoader 对象，布尔值is_train表示是否希望数据迭代器对象在每个迭代周期内打乱数据

batch_size=10
data_iter=load_array((features,labels),batch_size)

print(next(iter(data_iter)))
#DataLoader 是自动控制迭代的。在使用 DataLoader 加载数据集时，可以直接在 for 循环中使用它来迭代数据集的批次数据，而不需要手动调用 iter(...) 或 next(...) 函数,这里我们为了验证是否正常工作，先转换为iter，手动控制读取并打印第一个小批量样本

#第四步，定义模型
from torch import nn#import神经网络
net = nn.Sequential(nn.Linear(2, 1))#Sequential类将多个层串联在一起。 当给定输入数据时，Sequential实例将数据传入到第一层， 然后将第一层的输出作为第二层的输入，以此类推，nn.Linear(2, 1)表示从输入维度为 2 到输出维度为 1 的线性变换。即features是2维，输出的label是1维,它会按照output=input×weight+bias自动构建全连接层，

#第五步，初始化模型参数
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

#第六步，定义损失函数
loss=nn.MSELoss()#计算均方误差使用的是MSELoss类，也称为平方𝐿2范数, 默认情况下，它返回所有样本损失的平均值。

#第七步，定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)#生成一个优化器对象，net.parameters()返回net的全部待求参数，即上文的weight和bias,学习率0.03，

#第八步，训练
num_epochs=3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l=loss(net(X),y)#net是模型
        trainer.zero_grad()
        l.backward()#反向传递梯度
        trainer.step()#根据梯度更新参数
    l = loss(net(features), labels)#输出这一轮的误差
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
        








print('Hello World')
