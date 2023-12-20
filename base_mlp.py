import torch
import torch.nn as nn

class BaseMLPLayers(torch.module.nn):
    def __init__(self, weight_layer1, bias_layer1, weight_layer2, bias_layer2):
        super(BaseMLPLayers, self).__init__()
        self.linear_weight1 = nn.Parameter(weight_layer1)
        self.linear_bias1 = nn.Parameter(bias_layer1)
        self.linear_weight2 = nn.Parameter(weight_layer2)
        self.linear_bias2 = nn.Parameter(bias_layer2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        y1 = torch.matmul(x, self.linear_weight1) + self.linear_bias1
        y2 = self.relu(y1)
        y3 = torch.matmul(y2, self.linear_weight2) + self.linear_bias2
        return y3

