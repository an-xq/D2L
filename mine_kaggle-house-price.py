import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from torch.utils import data

#数据读取
# train_data=pd.read_csv('./data/kaggle_house_price/kaggle_house_pred_train.csv') #读入训练数据
# test_data=pd.read_csv('./data/kaggle_house_price/kaggle_house_pred_test.csv') #读入测试数据
train_data=pd.read_csv('./data/california-house-prices/train.csv') #读入训练数据
test_data=pd.read_csv('./data/california-house-prices/test.csv') #读入测试数据
# print(type(test_data))#验证
# print(train_data.head())
rows,cols=test_data.shape#验证
print(rows,"*",cols)
rows,cols=train_data.shape
print(rows,"*",cols)


#数据处理
labels = pd.DataFrame(train_data['Sold Price'])#记录训练数据的price
train_data=train_data.drop('Sold Price', axis=1)#并删除
rows,cols=test_data.shape#验证
print(rows,"*",cols)
rows,cols=train_data.shape
print(rows,"*",cols)

full_data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)#连接训练集和测试集
rows,cols=full_data.shape#验证
print(rows,"*",cols)
print(full_data.head(10))

full_data=full_data.drop(columns=full_data.columns[0])#去掉第一列id，防止过拟合

# float_column_names = data.select_dtypes(include=['float']).columns
dtypes = full_data.dtypes
non_string_columns = dtypes[dtypes != 'object'].index.tolist()#将数值列的NaN替换为这一列的平均值,并做标准化
for column in non_string_columns:
    if column in full_data.columns:
        mean_value = full_data[column].mean()  # 计算该列的平均值
        std_dev = full_data[column].std() # 计算该列的方差
        full_data[column].fillna(mean_value, inplace=True)  # 用平均值替换 NaN
        full_data[column] = (full_data[column] - mean_value) / std_dev
print(full_data.head(10))

full_data = pd.get_dummies(full_data,dtype=int)#将所有列独热编码
print(full_data.head(10))


# output_file_path = './data/california-house-prices/processed_data.csv'#将处理后的DataFrame导出到一个新的CSV文件，以供检查
# full_data.to_csv(output_file_path, index=False)
# print(f"\n处理后的 DataFrame 已保存到 '{output_file_path}'")

train_features = full_data.head(train_data.shape[0])
test_features = full_data.iloc[train_data.shape[0]:]
rows,cols=train_features.shape
print("train_features",rows,"*",cols)
rows,cols=test_features.shape
print("test_features",rows,"*",cols)

#数据迭代器生成
def data_loader(data_arrs,batch_size,is_train=True):
    dataset = data.TensorDataset(*data_arrs)#把features和labels分两个传入，声明一个TensorDataset对象
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

train_features_tensor = torch.tensor(train_features.values, dtype=torch.float32)
labels_tensor = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)  # 确保标签也变成2D张量
test_features_tensor = torch.tensor(test_features.values, dtype=torch.float32)

batchsize = 20
dataloader=data_loader([train_features_tensor,labels_tensor],batchsize)
# print("dataloader is ready")

#模型定义
net = nn.Sequential(
    nn.Linear(287,287*2),
    nn.ReLU(),
    nn.Linear(287*2,140),
    nn.ReLU(),
    nn.Linear(140,1)
    )

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_normal)

#损失函数
loss = nn.MSELoss()

def log_rmse(labels_hat, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    labels_hat=labels_hat+1
    clipped_preds = torch.clamp(labels_hat, min=1)

    if isinstance(labels, pd.DataFrame):
        labels = torch.tensor(labels.values, dtype=torch.float32)
#?

    labels = labels.squeeze()
    clipped_preds = clipped_preds.squeeze()

    log_clipped_preds = torch.log(clipped_preds)
    log_labels = torch.log(labels)

    return torch.sqrt(2 * loss(log_clipped_preds, log_labels).mean())

#优化器
lr=0.5
trainer = torch.optim.SGD(net.parameters(),lr=lr)

# initial_weight = net[0].weight.clone()  # 记录最初的权重
# initial_bias = net[0].bias.clone()       # 记录最初的偏置
# print("------------------")
# print(init_weights)
# print("------------------")
# print(initial_bias)
# print("------------------")

#训练
if __name__== '__main__':
    epochs = 4
    for epoch in range(epochs):
        # for X,y in dataloader:
        for i, (X, y) in enumerate(dataloader):
            l=log_rmse(net(X),y)
            # print(f"Epoch {epoch}, Batch {i}, Loss: {l.item():.6f}")
            trainer.zero_grad()
            l.backward()
            trainer.step()

            # if i % 10 == 0:  # 每10个batch打印一次
            #     # print(f"Epoch {epoch}, Batch {i}, Loss: {l.item():.6f}")
            #     first_layer = net[0]  # 假设nn.Sequential中的第一个层是[0]
            #     if isinstance(first_layer, nn.Linear):
            #         print(f"Weights Gradient Norm: {first_layer.weight.grad.norm().item()}")
            #         print(f"Bias Gradient Norm: {first_layer.bias.grad.norm().item()}")

        l=log_rmse(net(train_features_tensor),labels)
        print("epoch:{},l={:.6f}".format(epoch,l))

    final_data = pd.DataFrame(columns=['Id', 'Sold Price'])
    output_file_path = './data/california-house-prices/submission.csv'#将处理后的DataFrame导出到一个新的CSV文件，以供检查
    final_data['Id'] = test_data.iloc[:, 0]
    for index, row in test_data.iterrows():
        final_data.iloc[index, 0] = test_data.iloc[index, 0]
        # breakpoint()
        final_data.iloc[index, 1] = net(test_features_tensor[index,:]).item()
    final_data.to_csv(output_file_path, index=False)
    print(f"\n结果已保存 '{output_file_path}'")

    # w=model[0].weight.data
    # print(w.size())
    # print(w-w_true.reshape(w.size()).mean())
    # b=model[0].bias.data
    # # print(b.size())
    # print("w偏差:{:.6f}".format((w.reshape(w_true.size())-w_true).mean().item()))
    # print("b偏差:{:.6f}".format(b.item()-b_true))

#k折交叉验证？
    # # 检查训练后权重是否更新
    # final_weight = net[0].weight
    # final_bias = net[0].bias
    # print("Initial Weight:", initial_weight)
    # print("Final Weight:", final_weight)
    # print("Weight Change:", torch.sum(torch.abs(final_weight - initial_weight)).item())
    # print("Initial Bias:", initial_bias)
    # print("Final Bias:", final_bias)
    # print("Bias Change:", torch.sum(torch.abs(final_bias - initial_bias)).item())