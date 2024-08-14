import numpy as np
import torch

# x1=3
# x2=np.array([1,2,3])
# x3=np.zeros((5,2))
# for i in range(5):
#     x3[i][0]=i
#     x3[i][1]=i*i
# print(x3)
# x5=x3.sum(axis=1)
# print(x5)


# x4=np.arange(10)
# print(x4)

# print(np.power(x3,x2.reshape(1,-1)))

train_features=torch.normal(0,0.1,(2,2))
train_features[0][0]=1
train_features[0][1]=2
train_features[1][0]=3
train_features[1][1]=4

print(train_features[:,1]/2)