"""
基于物理约束的深度学习方法实现流体动力学建模
实现功能：
- 带物理约束的神经网络训练
- 多模型对比实验
- 流场可视化与定量分析
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

# ---------------------------- 模型定义 ----------------------------
class PhysicsInformedNN(torch.nn.Module):
    """物理信息神经网络架构
    参数：
        layers: 网络层结构配置，如 [3, 20, 20, 20, 4]
    """
    def __init__(self, layers: List[int]):
        super().__init__()
        self.linear_layers = torch.nn.ModuleList()
        
        # 初始化网络层
        for in_dim, out_dim in zip(layers[:-1], layers[1:]):
            layer = torch.nn.Linear(in_dim, out_dim)
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            self.linear_layers.append(layer)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor,...]:
        """前向传播
        返回：(u, v, p, phi) 的预测值
        """
        inputs = torch.cat([x, y, t], dim=1)
        for layer in self.linear_layers[:-1]:
            inputs = torch.tanh(layer(inputs))
        outputs = self.linear_layers[-1](inputs)
        return outputs.split(1, dim=1)

# ---------------------------- 训练模块 ----------------------------
def run_training() -> Dict[str, List[float]]:
    """执行完整训练流程
    返回：包含两种训练损失的字典
    """
    # 配置参数
    config = {
        'layers': [3, 20, 20, 20, 4],
        'epochs': 1000,
        'save_dir': 'models'
    }
    
    # 准备训练数据
    coords = generate_training_data(samples=100)
    
    # 初始化模型
    model_physics = PhysicsInformedNN(config['layers'])
    model_no_physics = PhysicsInformedNN(config['layers'])
    
    # 执行训练
    loss_history = {'physics': [], 'no_physics': []}
    train_model(model_physics, coords, config['epochs'], use_physics=True, loss_history=loss_history['physics'])
    train_model(model_no_physics, coords, config['epochs'], use_physics=False, loss_history=loss_history['no_physics'])
    
    # 保存模型
    os.makedirs(config['save_dir'], exist_ok=True)
    torch.save(model_physics.state_dict(), f"{config['save_dir']}/physics_model.pth")
    torch.save(model_no_physics.state_dict(), f"{config['save_dir']}/no_physics_model.pth")
    
    return loss_history

def generate_training_data(samples: int = 100) -> torch.Tensor:
    """生成训练坐标数据"""
    x = torch.rand(samples, 1) * 2 - 1  # [-1, 1] 均匀分布
    y = torch.rand(samples, 1) * 2 - 1
    t = torch.rand(samples, 1)          # [0, 1] 均匀分布
    return torch.cat([x, y, t], dim=1).requires_grad_(True)

def train_model(model: PhysicsInformedNN, 
               coords: torch.Tensor,
               epochs: int,
               use_physics: bool,
               loss_history: List[float]) -> None:
    """执行单个模型训练"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        u, v, p, phi = model(coords[:, 0:1], coords[:, 1:2], coords[:, 2:3])
        
        loss = calculate_loss(u, v, p, phi, coords, use_physics)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        print_progress(epoch, epochs, loss.item())

def calculate_loss(u: torch.Tensor,
                  v: torch.Tensor,
                  p: torch.Tensor,
                  phi: torch.Tensor,
                  coords: torch.Tensor,
                  use_physics: bool) -> torch.Tensor:
    """计算损失函数"""
    if use_physics:
        grad_outputs = torch.ones_like(p)
        p_grad = torch.autograd.grad(p, coords, grad_outputs=grad_outputs, 
                                    create_graph=True, retain_graph=True)[0]
        return (p_grad[:,0]**2 + p_grad[:,1]**2).mean()
    return torch.mean(u**2 + v**2 + p**2 + phi**2)

# ---------------------------- 可视化样式配置 ----------------------------
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'figure.dpi': 300,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

# ---------------------------- 改进后的可视化函数 ----------------------------
def plot_loss_comparison(loss_history: Dict[str, List[float]]) -> None:
    """训练损失对比曲线（优化样式）"""
    plt.figure(figsize=(8, 5))
    
    epochs = len(loss_history['physics'])
    x_ticks = np.linspace(0, epochs-1, 5, dtype=int)
    
    with plt.style.context('seaborn-v0_8-whitegrid'):  # 新版本兼容写法
        plt.semilogy(loss_history['physics'], 'r-', linewidth=1.5, label='Physics-informed')
        plt.semilogy(loss_history['no_physics'], 'b--', linewidth=1.5, label='Baseline')
        
        plt.xticks(x_ticks, [f"{x}" for x in x_ticks])
        plt.title(r'Training Loss Comparison ($\mathcal{L}_{phy}$ vs $\mathcal{L}_{data}$)')
        plt.xlabel('Epochs')
        plt.ylabel(r'Loss Value ($\log_{10}$)')
        plt.legend(frameon=True, shadow=True)
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()

def predict(input_coords, model_path='physics_model.pth'):
    """预测功能函数
    参数：
        input_coords: 形状为[N,3]的Tensor，包含[x,y,t]坐标
        model_path: 训练好的模型路径
    返回：
        u, v, p, phi 的预测值Tensor
    """
    # 加载模型
    model = PhysicsInformedNN([3, 20, 20, 20, 4])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 确保输入格式正确
    if not isinstance(input_coords, torch.Tensor):
        input_coords = torch.tensor(input_coords, dtype=torch.float32)
    
    # 进行预测
    with torch.no_grad():
        x = input_coords[:, 0:1]
        y = input_coords[:, 1:2]
        t = input_coords[:, 2:3]
        u, v, p, phi = model(x, y, t)
    
    return np.stack([u.numpy().squeeze(), 
                    v.numpy().squeeze(), 
                    p.numpy().squeeze(), 
                    phi.numpy().squeeze()], axis=1)

def visualize_predictions(coords, predictions):
    """可视化预测结果
    参数：
        coords: 输入坐标 [N,3]
        predictions: 形状为[N,4]的数组，包含(u, v, p, phi)
    """
    # 修改解包方式
    u = predictions[:, 0]
    v = predictions[:, 1] 
    p = predictions[:, 2]
    phi = predictions[:, 3]
    
    x = coords[:, 0]
    y = coords[:, 1]
    t = coords[:, 2]
    
    # 创建2x2的子图布局
    plt.figure(figsize=(15, 12))
    
    # 速度矢量图
    plt.subplot(2, 2, 1)
    plt.quiver(x, y, u, v, scale=30, width=0.002)
    plt.title('Velocity Vector Field')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # 压力场等高线
    plt.subplot(2, 2, 2)
    levels = np.linspace(p.min(), p.max(), 20)
    plt.tricontourf(x, y, p.squeeze(), levels=levels, cmap='jet')
    plt.colorbar(label='Pressure')
    plt.title('Pressure Contour')
    
    # 势函数分布
    plt.subplot(2, 2, 3)
    plt.scatter(x, y, c=phi, cmap='viridis', s=10)
    plt.colorbar(label='Phi')
    plt.title('Potential Function Distribution')
    
    # 时间演化（示例显示最后一个时间步）
    plt.subplot(2, 2, 4)
    time_indices = np.where(t == t.max())[0]
    plt.quiver(x[time_indices], y[time_indices], 
               u[time_indices], v[time_indices], 
               scale=30, width=0.002)
    plt.title(f'Velocity at t={t.max():.2f}')
    
    plt.tight_layout()
    plt.show()

def plot_comparison(loss_history):
    plt.figure(figsize=(10,6))
    plt.semilogy(loss_history['physics'], 'r-', label='With Physics')
    plt.semilogy(loss_history['no_physics'], 'b--', label='No Physics')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_comparison.png', dpi=300)
    plt.show()

def quantitative_analysis(physics_pred, no_physics_pred):
    """执行定量分析（包含多指标对比）
    参数：
        physics_pred: 带物理约束模型的预测结果 [N,4]
        no_physics_pred: 无物理约束模型的预测结果 [N,4]
    """
    # 生成模拟基准数据（理想旋转流场）
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    t = np.linspace(0, 1, 5)
    xx, yy, tt = np.meshgrid(x, y, t)
    
    # 基准流场定义
    ground_truth = np.stack([
        0.5 * yy.ravel(),   # u = 0.5y
        -0.5 * xx.ravel(),  # v = -0.5x 
        np.zeros_like(xx.ravel()),  # p=0
        np.sqrt(xx.ravel()**2 + yy.ravel()**2)  # φ = r
    ], axis=1)

    # 在quantitative_analysis函数开头添加形状检查
    print("预测结果形状验证:")
    print(f"physics_pred: {physics_pred.shape}")  # 应输出 (2000,4)
    print(f"no_physics_pred: {no_physics_pred.shape}")  # 应输出 (2000,4)
    print(f"ground_truth: {ground_truth.shape}")  # 应输出 (2000,4)

    # 计算各物理量MAE
    metrics = {
        'velocity': {
            'physics': np.mean(np.abs(physics_pred[:,:2] - ground_truth[:,:2])),
            'no_physics': np.mean(np.abs(no_physics_pred[:,:2] - ground_truth[:,:2]))
        },
        'pressure': {
            'physics': np.mean(np.abs(physics_pred[:,2] - ground_truth[:,2])),
            'no_physics': np.mean(np.abs(no_physics_pred[:,2] - ground_truth[:,2]))
        },
        'potential': {
            'physics': np.mean(np.abs(physics_pred[:,3] - ground_truth[:,3])),
            'no_physics': np.mean(np.abs(no_physics_pred[:,3] - ground_truth[:,3]))
        }
    }

    # 打印格式化表格
    print("""
    Quantitative Analysis Results:
    +----------------------+----------------+-------------------+
    |       Metric         | With Physics   | Without Physics   |
    +----------------------+----------------+-------------------+
    | Velocity MAE         | {velocity_phy:.4f}      | {velocity_nophy:.4f}       |
    | Pressure MAE         | {pressure_phy:.4f}      | {pressure_nophy:.4f}       | 
    | Potential MAE        | {potential_phy:.4f}      | {potential_nophy:.4f}       |
    +----------------------+----------------+-------------------+
    """.format(
        velocity_phy=metrics['velocity']['physics'],
        velocity_nophy=metrics['velocity']['no_physics'],
        pressure_phy=metrics['pressure']['physics'],
        pressure_nophy=metrics['pressure']['no_physics'],
        potential_phy=metrics['potential']['physics'],
        potential_nophy=metrics['potential']['no_physics']
    ))

    # 返回指标用于后续可视化
    return metrics

def plot_velocity_field(coords, predictions):
    """可视化速度矢量场
    参数：
        coords: 输入坐标 [N,3]
        predictions: 预测结果 [N,4]
    """
    # 确保predictions是二维数组
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 4)
        
    x = coords[:, 0]
    y = coords[:, 1]
    u = predictions[:, 0]  # 现在可以安全索引
    v = predictions[:, 1]
    
    plt.quiver(x, y, u, v, 
               scale=30, 
               width=0.002,
               angles='xy',
               scale_units='xy')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Velocity Vector Field')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_flow_comparison(coords: np.ndarray, 
                        physics_pred: np.ndarray, 
                        no_physics_pred: np.ndarray,
                        ground_truth: np.ndarray) -> None:
    """流场对比可视化（专业学术样式）"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    titles = [
        (r'Ground Truth $\mathbf{u} = (0.5y, -0.5x)$', ground_truth),
        ('Baseline Prediction', no_physics_pred),
        ('Physics-informed Prediction', physics_pred)
    ]
    
    # 统一颜色映射范围
    speed_min = min(np.linalg.norm(ground_truth[:,:2], axis=1).min(),
                   np.linalg.norm(no_physics_pred[:,:2], axis=1).min(),
                   np.linalg.norm(physics_pred[:,:2], axis=1).min())
    speed_max = max(np.linalg.norm(ground_truth[:,:2], axis=1).max(),
                   np.linalg.norm(no_physics_pred[:,:2], axis=1).max(),
                   np.linalg.norm(physics_pred[:,:2], axis=1).max())
    
    for ax, (title, data), cmap in zip(axes, titles, ['viridis', 'plasma', 'plasma']):
        # 速度矢量与模长叠加显示
        speed = np.linalg.norm(data[:,:2], axis=1)
        stream = ax.scatter(coords[:,0], coords[:,1], c=speed, 
                          cmap=cmap, s=15, alpha=0.8, 
                          vmin=speed_min, vmax=speed_max)
        
        # 添加矢量箭头
        skip = (slice(None, None, 8), slice(None, None, 8))
        ax.quiver(coords[::8,0], coords[::8,1], 
                 data[::8,0], data[::8,1],
                 scale=50, width=0.003, headwidth=3)
        
        ax.set_title(title, pad=15)
        ax.set_xlabel(r'$x$ coordinate')
        ax.set_ylabel(r'$y$ coordinate', labelpad=10)
        ax.set_aspect('equal')
        
        # 添加颜色条
        cbar = plt.colorbar(stream, ax=ax, shrink=0.8)
        cbar.set_label(r'Speed Magnitude ($||\mathbf{u}||_2$)', rotation=270, labelpad=15)
    
    plt.tight_layout(pad=2.0)

def compute_continuity_residual(coords, predictions):
    """计算质量守恒残差 ∂u/∂x + ∂v/∂y"""
    # 转换为Tensor并启用梯度
    coords_tensor = torch.tensor(coords, dtype=torch.float32, requires_grad=True)
    u = torch.tensor(predictions[:,0], dtype=torch.float32)
    v = torch.tensor(predictions[:,1], dtype=torch.float32)
    
    # 计算u对x的梯度（添加allow_unused=True）
    u_grad = torch.autograd.grad(
        outputs=u,
        inputs=coords_tensor,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        allow_unused=True  # 允许未使用的梯度
    )[0][:,0] if coords_tensor.grad is not None else torch.zeros_like(coords_tensor[:,0])
    
    # 计算v对y的梯度（添加allow_unused=True）
    v_grad = torch.autograd.grad(
        outputs=v,
        inputs=coords_tensor,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        allow_unused=True  # 允许未使用的梯度
    )[0][:,1] if coords_tensor.grad is not None else torch.zeros_like(coords_tensor[:,1])
    
    # 处理可能的None值
    u_grad = u_grad if u_grad is not None else torch.zeros_like(coords_tensor[:,0])
    v_grad = v_grad if v_grad is not None else torch.zeros_like(coords_tensor[:,1])
    
    return (u_grad + v_grad).detach().numpy()

def plot_residual_distributions(coords: np.ndarray,
                               physics_pred: np.ndarray,
                               no_physics_pred: np.ndarray) -> None:
    """残差分布可视化（增强统计信息）"""
    res_physics = compute_continuity_residual(coords, physics_pred)
    res_no_physics = compute_continuity_residual(coords, no_physics_pred)
    
    fig = plt.figure(figsize=(12, 5))
    bins = np.linspace(min(res_no_physics.min(), res_physics.min()),
                      max(res_no_physics.max(), res_physics.max()), 50)
    
    # 残差统计指标
    stats_phy = (r'$\mu={:.2e}$'.format(np.mean(res_physics)), 
                r'$\sigma={:.2e}$'.format(np.std(res_physics)))
    stats_base = (r'$\mu={:.2e}$'.format(np.mean(res_no_physics)), 
                 r'$\sigma={:.2e}$'.format(np.std(res_no_physics)))
    
    # 绘制直方图
    ax1 = plt.subplot(121)
    ax1.hist(res_no_physics, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    ax1.set_title('Baseline Model Residuals\n' + '\n'.join(stats_base))
    ax1.set_xlabel(r'Continuity Residual ($\nabla \cdot \mathbf{u}$)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(122, sharey=ax1)
    ax2.hist(res_physics, bins=bins, color='red', alpha=0.7, edgecolor='black')
    ax2.set_title('Physics-informed Model Residuals\n' + '\n'.join(stats_phy))
    ax2.set_xlabel(r'Continuity Residual ($\nabla \cdot \mathbf{u}$)')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Divergence Residual Distribution Comparison', y=1.02)
    plt.tight_layout()

# ---------------------------- 工具函数 ----------------------------
def print_progress(epoch: int, total: int, loss: float) -> None:
    """打印训练进度"""
    if epoch % 100 == 0 or epoch == total-1:
        print(f"Epoch {epoch:04d}/{total} | Loss: {loss:.4e}")

def generate_test_coords(samples: int = 20) -> np.ndarray:
    """生成测试坐标数据（结构化网格）"""
    x = np.linspace(-1, 1, samples)
    y = np.linspace(-1, 1, samples)
    t = np.linspace(0, 1, 5)  # 5个时间步
    xx, yy, tt = np.meshgrid(x, y, t)
    return np.vstack([xx.ravel(), yy.ravel(), tt.ravel()]).T

# ---------------------------- 可视化模块 ----------------------------
def visualize_results(loss_history: Dict[str, List[float]],
                     test_coords: np.ndarray,
                     physics_pred: np.ndarray,
                     no_physics_pred: np.ndarray,
                     ground_truth: np.ndarray) -> None:
    """整合多个可视化内容
    参数：
        loss_history: 训练损失记录
        test_coords: 测试坐标数据
        physics_pred: 物理约束模型预测结果
        no_physics_pred: 无约束模型预测结果
        ground_truth: 基准真值数据
    """
    plt.figure(figsize=(15, 10))
    
    # 损失对比
    plt.subplot(2, 2, 1)
    plot_loss_comparison(loss_history)
    
    # 流场对比
    plt.subplot(2, 2, 2)
    plot_flow_comparison(test_coords, physics_pred, no_physics_pred, ground_truth)
    
    # 残差分析
    plt.subplot(2, 1, 2)
    plot_residual_distributions(test_coords, physics_pred, no_physics_pred)
    
    plt.tight_layout()
    plt.savefig('combined_analysis.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # 执行完整流程
    loss_history = run_training()
    
    # 生成测试数据（使用新函数）
    test_coords = generate_test_coords()
    physics_pred = predict(test_coords, model_path='models/physics_model.pth')
    no_physics_pred = predict(test_coords, model_path='models/no_physics_model.pth')
    
    # 生成基准真值
    x = test_coords[:, 0]
    y = test_coords[:, 1]
    ground_truth = np.stack([
        0.5 * y,
        -0.5 * x,
        np.zeros_like(x),
        np.sqrt(x**2 + y**2)
    ], axis=1)
    
    # 分析与可视化
    visualize_results(
        loss_history,
        test_coords,
        physics_pred,
        no_physics_pred,
        ground_truth
    )
    quantitative_analysis(physics_pred, no_physics_pred)

    # 验证函数存在性
    assert 'visualize_results' in globals(), "可视化函数未正确定义"
    print("所有可视化函数已正确加载")

# 测试代码
test_data = generate_test_coords()
print(f"测试数据形状：{test_data.shape}")  # 应输出 (2000, 3)
print(f"坐标范围：x({test_data[:,0].min():.2f}-{test_data[:,0].max():.2f})")
print(f"          y({test_data[:,1].min():.2f}-{test_data[:,1].max():.2f})")
print(f"          t({test_data[:,2].min():.2f}-{test_data[:,2].max():.2f})")

print(plt.style.available)  # 查看所有可用样式名称



