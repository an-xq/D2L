import torch
import os
import pandas as pd

os.makedirs(os.path.join('.','data'),exist_ok=True)
data_file = os.path.join('.','data','house_tiny.csv')
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127\n')
    f.write('2,NA,106\n')
    f.write('4,NA,178\n')
    f.write('NA,NA,140\n')

data = pd.read_csv(data_file)
print(data)

inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)

inputs = pd.get_dummies(inputs,dummy_na=True)
inputs = inputs*1
print(inputs)

x,y=torch.tensor(inputs.values),torch.tensor(outputs.values)
print(x,y)

print(x.ndim)

# A = torch.arange(24).reshape(2,4,3)
# print("A:",A)
# B = A.sum(dim=0)
# E = A.sum(dim=0,keepdim=True)
# # C = A.sum(dim=[0,1])
# # D = A.sum(dim=2)
# print("B:",B)
# print("E:",E)
# print("C:",C)
# print("D:",D)

# A = torch.ones(24).reshape(2,4,3)
# B = A.cumsum(dim=2)
# print("B:",B)

A = torch.arange(12).reshape(2,2,3)
B = A.sum(dim=[1,2])
print("a:",A)
print("B:",B)