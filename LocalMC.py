import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# 定义前馈神经网络
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义损失函数为平方差损失函数
def custom_loss(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    regularization = torch.mean(y_pred ** 2) * 0.01
    return mse + regularization

csv_file = 'harmonic_oscillator_trajectory.csv'
data = pd.read_csv(csv_file)

# 提取相空间位置和动量
X = data[['Position (x)', 'Momentum (p)']].values

print(X.shape)

# 初始化模型、优化器
model = FeedForwardNN(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    y_pred = model(X_train)
    loss = custom_loss(y_train, y_pred)
    
    # 反向传播
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
