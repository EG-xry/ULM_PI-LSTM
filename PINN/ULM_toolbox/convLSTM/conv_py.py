import sys
import subprocess


# 自动安装matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("正在自动安装matplotlib...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt


# 自动安装必要库
required = {'numpy', 'torch', 'matplotlib'}
try:
    import numpy as np
    import torch
    import torch.nn as nn
except ImportError as e:
    missing = e.name if e.name else [pkg for pkg in required if not __import__(pkg)]
    print(f"正在安装缺失的依赖: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    import numpy as np
    import torch
    import torch.nn as nn

# 生成合成数据（移动的白色方块）
def generate_data(num_samples=100, seq_len=5, size=32):
    data = torch.zeros(num_samples, seq_len+1, 1, size, size)
    for i in range(num_samples):
        x, y = np.random.randint(0, size-8, 2)
        dx, dy = np.random.randint(-3, 4, 2)
        for t in range(seq_len+1):
            data[i, t, 0, y:y+8, x:x+8] = 1.0
            x = np.clip(x + dx, 0, size-8)
            y = np.clip(y + dy, 0, size-8)
    return data[:, :-1], data[:, 1:]

# ConvLSTM 单元实现
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        
        combined = torch.cat([x, h_prev], dim=1)  # 拼接输入和隐藏状态
        gates = self.conv(combined)
        
        # 分割输入门、遗忘门、输出门和候选记忆
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

# 完整的 ConvLSTM 模型
class ConvLSTM(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=16, num_layers=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        self.conv_lstm_cell = ConvLSTMCell(in_channels, hidden_channels)
        self.conv = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        device = x.device
        batch_size, seq_len, _, h, w = x.size()
        
        # 初始化隐藏状态
        h_t = torch.zeros(batch_size, self.hidden_channels, h, w).to(device)
        c_t = torch.zeros(batch_size, self.hidden_channels, h, w).to(device)
        
        outputs = []
        for t in range(seq_len):
            h_t, c_t = self.conv_lstm_cell(x[:, t], (h_t, c_t))
            outputs.append(h_t.unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)
        return self.conv(output[:, -1])  # 预测最后一帧

# 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvLSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 生成数据
train_input, train_target = generate_data(100, seq_len=5)
train_input = train_input.to(device)
train_target = train_target.to(device)

# 训练循环
losses = []
for epoch in range(50):
    optimizer.zero_grad()
    output = model(train_input)
    loss = criterion(output, train_target[:, -1])
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 可视化结果
plt.plot(losses)
plt.title("Training Loss")
plt.show()

# 测试预测
test_input, test_target = generate_data(1, seq_len=5)
with torch.no_grad():
    pred = model(test_input.to(device)).cpu()

fig, axes = plt.subplots(1, 3, figsize=(12,4))
axes[0].imshow(test_input[0, -1, 0].cpu(), cmap='gray')
axes[0].set_title("Input Frame")
axes[1].imshow(pred[0, 0], cmap='gray', vmin=0, vmax=1)
axes[1].set_title("Prediction")
axes[2].imshow(test_target[0, -1, 0].cpu(), cmap='gray')
axes[2].set_title("Target")
plt.show()