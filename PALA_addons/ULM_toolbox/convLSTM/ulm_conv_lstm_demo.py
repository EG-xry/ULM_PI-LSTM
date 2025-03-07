import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# 第一部分：生成模拟数据
def generate_simulated_data(img_size=(128, 128), num_frames=100, num_particles=50):
    np.random.seed(2023)
    true_tracks = []
    coords_seq = []
    
    # 初始化粒子的位置和速度
    start_pos = np.random.rand(num_particles, 2) * img_size
    velocities = np.random.randn(num_particles, 2) * 0.8
    
    # 生成各帧坐标
    for t in range(num_frames):
        current_coords = []
        for p in range(num_particles):
            if t == 0:
                pos = start_pos[p]
            else:
                # 更新速度并添加噪声
                velocities[p] += np.random.randn(2) * 0.2
                pos = true_tracks[p][t-1][1:] + velocities[p]
                pos = np.clip(pos, 1, img_size)
            
            # 保存真实轨迹
            if t == 0:
                true_tracks.append([(t+1, *pos)])
            else:
                true_tracks[p].append((t+1, *pos))
            
            current_coords.append(pos + np.random.randn(2)*0.5)
        
        coords_seq.append(np.array(current_coords))
    
    return coords_seq, true_tracks

# 新增 ConvLSTM 单元实现
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=32, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        # 添加通道数验证
        assert in_channels == 1, "输入通道必须为1（热力图）"
        assert hidden_channels == 32, "隐藏通道必须为32"
        
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, hidden):
        # 添加维度调试信息
        print(f"输入维度进入LSTM Cell: {x.shape}")
        print(f"隐藏状态维度: {hidden[0].shape}")
        
        # 修复维度处理逻辑
        if x.dim() == 5:
            x = x.squeeze(1)  # [batch, 1, C, H, W] → [batch, C, H, W]
        elif x.dim() == 4:
            x = x.unsqueeze(1)  # [batch, C, H, W] → [batch, 1, C, H, W]
        
        # 保持原有断言
        assert x.size(1) == 1, f"输入通道应为1，实际为{x.size(1)}"
        
        h_prev, c_prev = hidden
        
        # 强制通道数验证
        assert x.size(1) == 1, f"输入通道应为1，实际为{x.size(1)}"
        assert h_prev.size(1) == 32, f"隐藏通道应为32，实际为{h_prev.size(1)}"
        
        combined = torch.cat([x, h_prev], dim=1)  # 正确通道数 1+32=33
        gates = self.conv(combined)
        
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# 修改后的跟踪器类
class ConvLSTMTracker(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.hidden_dim = 32
        
        # 添加初始化验证
        assert self.hidden_dim == 32, "隐藏维度必须为32"
        
        self.conv_lstm_cell = ConvLSTMCell(
            in_channels=1,
            hidden_channels=self.hidden_dim
        )
        
        # 输出层通道验证
        self.conv = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        assert self.conv.in_channels == 32, "输出层输入通道错误"
        
    def coords_to_heatmap(self, coords_seq):
        heatmaps = np.zeros((len(coords_seq), 1, *self.img_size))
        for t, coords in enumerate(coords_seq):
            heatmap = np.zeros(self.img_size)
            for x, y in coords:
                if 0 <= x < self.img_size[1] and 0 <= y < self.img_size[0]:
                    heatmap[int(y-1), int(x-1)] = 1
            heatmaps[t] = gaussian_filter(heatmap, sigma=1.5)
        return torch.FloatTensor(heatmaps)
    
    def track(self, coords_seq, seq_length=5):
        # 修复输入处理逻辑（关键修改）
        heatmap_seq = self.coords_to_heatmap(coords_seq)  # 形状应为 [T, 1, H, W]
        
        # 添加热力图通道验证
        assert heatmap_seq.size(1) == 1, f"热力图通道数异常，应为1，实际为{heatmap_seq.size(1)}"
        
        # 修正维度处理
        inputs = heatmap_seq[t-seq_length:t]  # 保持 [seq_length, 1, H, W]
        inputs = inputs.permute(1, 0, 2, 3)  # [1, seq_length, H, W]
        
        # 添加输入验证
        print(f"输入维度验证: {inputs.shape}")  # 应为 [1, seq_length, H, W]
        assert inputs.size(0) == 1, f"输入通道数异常，应为1，实际为{inputs.size(0)}"
        
        # 添加运行时验证
        assert self.hidden_dim == 32, "运行时隐藏维度异常"
        
        device = next(self.parameters()).device
        heatmap_seq = heatmap_seq.unsqueeze(1).to(device)  # [T, C, H, W]
        
        # 初始化隐藏状态
        h = torch.zeros(1, 32, *self.img_size).to(device)
        c = torch.zeros(1, 32, *self.img_size).to(device)
        print(f"隐藏状态通道验证: {h.shape}")  # 应为[1,32,H,W]
        
        # 存储所有预测偏移量
        all_offsets = []
        for t in range(len(coords_seq)):
            if t < seq_length:
                continue
                
            # 处理序列时保持维度一致
            for s in range(seq_length):
                x_t = inputs[:, s].unsqueeze(1)  # [C, 1, H, W]
                h_t, c_t = h, c
                h_t, c_t = self.conv_lstm_cell(x_t, (h_t, c_t))
            
            # 预测偏移量
            offset = self.conv(h_t)
            all_offsets.append(offset)
        
        # 重建轨迹（修改部分）
        tracks = []
        for p in range(len(coords_seq[0])):
            track = []
            for t in range(len(coords_seq)):
                if t < seq_length:
                    track.append(coords_seq[t][p])
                else:
                    offset_idx = t - seq_length
                    pred_offset = all_offsets[offset_idx][0, :, int(coords_seq[t-1][p][1]), int(coords_seq[t-1][p][0])]
                    pred_pos = coords_seq[t-1][p] + pred_offset.cpu().numpy()
                    track.append(pred_pos)
            tracks.append(np.array(track))
        return tracks

# 第三部分：训练准备
def prepare_sequence_data(heatmap_seq, seq_length):
    num_frames = heatmap_seq.shape[0]
    X = []
    Y = []
    for i in range(num_frames - seq_length):
        X.append(heatmap_seq[i:i+seq_length])
        Y.append(heatmap_seq[i+seq_length] - heatmap_seq[i+seq_length-1])
    return torch.stack(X), torch.stack(Y)

# 第四部分：主程序
def main():
    # 生成数据
    img_size = (128, 128)
    coords_seq, true_tracks = generate_simulated_data(img_size=img_size)
    
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 新增设备定义
    
    # 初始化跟踪器
    tracker = ConvLSTMTracker(img_size).to(device)  # 移动到设备
    
    # 准备训练数据
    heatmap_seq = tracker.coords_to_heatmap(coords_seq)
    X_train, Y_train = prepare_sequence_data(heatmap_seq, seq_length=5)
    
    # 训练配置
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(tracker.parameters(), lr=1e-4)
    
    # 修改后的训练循环（修复device未定义）
    losses = []
    for epoch in range(30):
        epoch_loss = 0
        for i in range(len(X_train)):
            inputs = X_train[i].permute(1, 0, 2, 3)  # [C, T, H, W] → [T, C, H, W]
            inputs = inputs.unsqueeze(0)  # 添加batch维度 [1, T, C, H, W]
            
            # 初始化隐藏状态
            h = torch.zeros(1, tracker.hidden_dim, *img_size).to(device)
            c = torch.zeros(1, tracker.hidden_dim, *img_size).to(device)
            
            # 按时间步处理
            for s in range(inputs.size(1)):
                h, c = tracker.conv_lstm_cell(inputs[:, s], (h, c))
            
            output = tracker.conv(h)
            
            target = Y_train[i].unsqueeze(0).to(device)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(X_train)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    
    # 测试跟踪
    test_coords = coords_seq[:50]
    pred_tracks = tracker.track(test_coords)
    
    # 可视化结果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    for track in true_tracks[:50]:
        track = np.array(track)
        plt.plot(track[:,1], track[:,2], lw=1)
    plt.title('True Tracks')
    plt.xlim(0, img_size[1])
    plt.ylim(0, img_size[0])
    
    plt.subplot(1, 2, 2)
    for track in pred_tracks[:50]:
        plt.plot(track[:,0], track[:,1], lw=1)
    plt.title('Predicted Tracks')
    plt.xlim(0, img_size[1])
    plt.ylim(0, img_size[0])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 