import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init

class Dense(nn.Module):
    def __init__(self, in_dims, out_dims, nonlin="relu", merge=False, dropout=0.0, trainable=True, **kwargs):

        super(Dense, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.nonlin = nonlin
        self.dropout = dropout
        self.trainable = trainable
        self.merge = merge
        self.device = torch.device("cuda")

        self.W = nn.Parameter(init.kaiming_uniform_(torch.randn(in_dims, out_dims, requires_grad=self.trainable)).type(torch.float).to(self.device))
        self.b = nn.Parameter(torch.zeros(out_dims, requires_grad=self.trainable).type(torch.float).to(self.device))

        self.drop_layer = nn.Dropout(p=self.dropout)


    def forward(self, x):
        out_dims = self.in_dims if self.out_dims is None else self.out_dims

        Z = torch.matmul(x, self.W) + self.b
        if self.nonlin == "relu":
            Z = F.relu(Z)

        if self.nonlin == "sigmoid":
            Z = torch.sigmoid(Z)

        if self.merge:
            Z = torch.sum(Z, 0)

        if self.dropout and self.nonlin != "linear" and self.nonlin != "sigmoid":
            Z = self.drop_layer(Z)

        return Z