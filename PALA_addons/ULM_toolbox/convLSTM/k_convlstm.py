import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 数据生成函数
def generate_moving_block(num_samples=100, seq_len=10, image_size=(64, 64)):
    """生成移动白色方块的序列数据"""
    data = np.zeros((num_samples, seq_len, image_size[0], image_size[1], 1))
    for i in range(num_samples):
        # 随机初始位置
        x, y = np.random.randint(0, image_size[0]-8), np.random.randint(0, image_size[1]-8)
        
        for t in range(seq_len):
            # 重置位置（演示不同运动模式）
            if t == 0 or t == seq_len//2:
                x = np.random.randint(0, image_size[0]-8)
                y = np.random.randint(0, image_size[1]-8)
                
            # 绘制8x8白色方块
            data[i, t, x:x+8, y:y+8, 0] = 1.0
            
            # 随机移动方向
            x += np.random.choice([-1, 0, 1])
            y += np.random.choice([-1, 0, 1])
            
            # 边界限制
            x = np.clip(x, 0, image_size[0]-8)
            y = np.clip(y, 0, image_size[1]-8)
    return data

# 生成数据（100个样本，每个包含10帧）
data = generate_moving_block(num_samples=100)
train_X = data[:, :5]  # 前5帧作为输入
train_y = data[:, 5:]  # 后5帧作为预测目标

class ConvLSTM2d(nn.Module):
    """PyTorch ConvLSTM实现"""
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2  # 保持尺寸不变
        
        # 门控卷积层
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,  # 输入/遗忘/输出/候选门
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, hidden_state=None):
        batch, _, height, width = x.size()
        
        # 初始化隐藏状态
        if hidden_state is None:
            h = torch.zeros(batch, self.hidden_dim, height, width, device=x.device)
            c = torch.zeros_like(h)
        else:
            h, c = hidden_state
        
        # 拼接输入和隐藏状态
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        
        # 分割门控信号
        input_gate, forget_gate, output_gate, candidate_gate = gates.chunk(4, dim=1)
        
        # 计算新状态
        i = torch.sigmoid(input_gate)
        f = torch.sigmoid(forget_gate)
        o = torch.sigmoid(output_gate)
        g = torch.tanh(candidate_gate)
        
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        
        return new_h, (new_h, new_c)

class ConvLSTMModel(nn.Module):
    """完整的ConvLSTM预测模型"""
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super().__init__()
        self.convlstm = ConvLSTM2d(input_dim, hidden_dim)
        self.final_conv = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x形状: (batch, seq_len, C, H, W)
        batch, seq_len, _, height, width = x.size()
        x = x.permute(1, 0, 2, 3, 4)  # (seq_len, batch, C, H, W)
        
        h, c = None, None
        for t in range(seq_len):
            h, (h, c) = self.convlstm(x[t], (h, c) if h is not None else None)
        
        return self.final_conv(h)

# 生成测试数据
def generate_moving_square(seq_length=5, size=64, batch_size=4):
    """生成移动方块的序列数据"""
    sequences = torch.zeros(batch_size, seq_length, 1, size, size)
    for b in range(batch_size):
        start = np.random.randint(0, size-20)
        for t in range(seq_length):
            x = start + t*5
            sequences[b, t, 0, x:x+10, x:x+10] = 1.0
    return sequences

# 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvLSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(10):
    # 生成数据
    inputs = generate_moving_square().to(device)
    targets = torch.roll(inputs[:, -1], shifts=5, dims=2)  # 目标为最后一帧右移5像素
    
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 可视化结果
with torch.no_grad():
    test_input = generate_moving_square(batch_size=1).to(device)
    prediction = model(test_input)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(test_input[0, -1, 0].cpu(), cmap='gray')
    plt.title("Last Input Frame")
    
    plt.subplot(132)
    plt.imshow(torch.roll(test_input[0, -1, 0], shifts=5, dims=1).cpu(), cmap='gray')
    plt.title("Ground Truth")
    
    plt.subplot(133)
    plt.imshow(prediction[0, 0].cpu(), cmap='gray')
    plt.title("Prediction")
    plt.show()
