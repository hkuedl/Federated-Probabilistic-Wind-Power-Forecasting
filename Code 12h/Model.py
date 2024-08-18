from torch import nn
from torch.nn import functional as F
import torch

class ANN(nn.Module):
    def __init__(self, args):
        super(ANN, self).__init__()
        self.args = args
        self.hidden_layers = nn.ModuleList()
        self.relu = nn.ReLU()
        # 添加输入层到第一个隐藏层
        self.hidden_layers.append(nn.Linear(self.args.input_size, self.args.hidden_layers[0]))

        # 添加隐藏层
        for k in range(len(self.args.hidden_layers)-1):
            self.hidden_layers.append(nn.Linear(self.args.hidden_layers[k], self.args.hidden_layers[k+1]))

        # 添加输出层
        self.output = nn.Linear(self.args.hidden_layers[-1], self.args.output_size)

    def forward(self, x):
        # 通过隐藏层
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output(x)
        return x