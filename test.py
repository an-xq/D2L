import numpy as np
import torch

def dropout(X,p):
    assert 0<=p<=1
    if p==1:
        return X
    if p==0:
        return torch.zeros_like(X)
    temp=torch.randn(X.shape)
    m=(temp>p).float()
    return X*m /(1.0-p)

X=torch.arange(16).reshape((2, 8))
y=dropout(X,0.5)
print(X)
print(y)