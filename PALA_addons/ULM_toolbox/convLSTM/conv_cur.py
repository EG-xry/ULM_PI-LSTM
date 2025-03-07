import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成合成数据：移动的圆形 + 对应曲线
def generate_image_curve_data(num_samples=100, seq_len=10, img_size=64):
    images = np.zeros((num_samples, seq_len, 1, img_size, img_size))
    curves = np.zeros((num_samples, seq_len, 1))
    
    for i in range(num_samples):
        # 生成随机运动参数
        start_x = np.random.randint(20, img_size-20)
        speed = np.random.uniform(-3, 3)
        
        for t in range(seq_len):
            # 当前圆心x坐标
            x = start_x + int(t * speed)
            x = np.clip(x, 10, img_size-10)
            
            # 生成图像
            y = img_size // 2
            xx, yy = np.mgrid[:img_size, :img_size]
            circle = ((xx - x)**2 + (yy - y)**2) < 100
            images[i, t, 0] = circle.astype(float)
            
            # 生成对应曲线值（标准化到0-1）
            curves[i, t, 0] = x / img_size
    
    return torch.FloatTensor(images), torch.FloatTensor(curves)

# ConvLSTM模型架构
class ImageToCurveLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器（处理图像序列）
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # ConvLSTM层
        self.lstm = nn.LSTM(
            input_size=32*(64//4)**2,  # 根据池化层计算
            hidden_size=128,
            batch_first=True
        )
        
        # 解码器（预测曲线）
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 编码处理
        encoded = []
        for t in range(seq_len):
            img = x[:, t]  # [B, C, H, W]
            feat = self.encoder(img)
            feat = feat.view(batch_size, -1)
            encoded.append(feat)
        
        encoded = torch.stack(encoded, dim=1)  # [B, T, D]
        
        # LSTM处理
        lstm_out, _ = self.lstm(encoded)
        
        # 解码预测
        outputs = []
        for t in range(seq_len):
            output = self.decoder(lstm_out[:, t])
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)

# 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageToCurveLSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 生成数据
images, curves = generate_image_curve_data(200, seq_len=10)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(images, curves),
    batch_size=32,
    shuffle=True
)

# 训练循环
losses = []
for epoch in range(50):
    epoch_loss = 0
    for batch_imgs, batch_curves in train_loader:
        batch_imgs = batch_imgs.to(device)
        batch_curves = batch_curves.to(device)
        
        optimizer.zero_grad()
        preds = model(batch_imgs)
        loss = criterion(preds, batch_curves)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# 可视化结果
plt.figure(figsize=(15, 5))

# 训练损失曲线
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")

# 样本预测对比
with torch.no_grad():
    test_imgs, test_curves = generate_image_curve_data(3, seq_len=10)
    pred_curves = model(test_imgs.to(device)).cpu()

for i in range(3):
    # 显示首尾帧图像
    plt.subplot(3, 3, i*3 + 2)
    plt.imshow(test_imgs[i, 0, 0], cmap='gray')
    plt.title(f"Sample {i+1}\nFirst Frame")
    plt.axis('off')
    
    plt.subplot(3, 3, i*3 + 3)
    plt.imshow(test_imgs[i, -1, 0], cmap='gray')
    plt.title("Last Frame")
    plt.axis('off')
    
    # 显示曲线对比
    plt.subplot(3, 3, i*3 + 1)
    plt.plot(test_curves[i, :, 0], 'b-', label='True')
    plt.plot(pred_curves[i, :, 0], 'r--', label='Predicted')
    plt.legend()
    plt.title("Position Curve")

plt.tight_layout()
plt.show()
                        