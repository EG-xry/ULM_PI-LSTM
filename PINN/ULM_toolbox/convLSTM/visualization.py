import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 在导入任何科学计算库之前设置

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D

def plot_sequence(sequence, title="Input Sequence"):
    """可视化输入序列"""
    plt.figure(figsize=(12, 3))
    for i in range(min(5, sequence.shape[1])):  # 显示前5个时间步
        plt.subplot(1, 5, i+1)
        plt.imshow(sequence[0, i, ..., 0], cmap='viridis')
        plt.title(f"t={i}")
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_prediction(input_seq, target, prediction, epoch=None):
    """修复形状的预测结果对比"""
    # 确保数据维度正确
    def prepare_img(arr):
        arr = arr.squeeze()  # 去除单一维度
        if arr.ndim == 3:    # 处理批次维度
            return arr[0]
        return arr         
    
    plt.figure(figsize=(10, 4))
    
    # 最后一个输入帧
    plt.subplot(1, 3, 1)
    last_input = prepare_img(input_seq[0, -1])
    plt.imshow(last_input, cmap='viridis')
    plt.title(f"Input\nShape: {last_input.shape}")
    
    # 真实目标
    plt.subplot(1, 3, 2)
    target_img = prepare_img(target[0])
    plt.imshow(target_img, cmap='viridis')
    plt.title(f"Target\nShape: {target_img.shape}")
    
    # 预测结果
    plt.subplot(1, 3, 3)
    pred_img = prepare_img(prediction[0])
    plt.imshow(pred_img, cmap='viridis')
    plt.title(f"Prediction\nShape: {pred_img.shape}")
    
    if epoch is not None:
        plt.suptitle(f"Epoch {epoch+1}")
    plt.tight_layout()
    plt.show()

def generate_moving_square(seq_length=5, size=64, batch_size=1):
    """生成正确维度的移动方块"""
    seq = np.zeros((batch_size, seq_length, size, size, 1))
    for b in range(batch_size):
        square_size = 10
        start = np.random.randint(0, size-square_size-1)
        for t in range(seq_length):
            x = start + t * (size - square_size - start) // max(1, seq_length-1)
            y = start
            seq[b, t, y:y+square_size, x:x+square_size, 0] = 1.0
    return seq

def generate_target(sequence, shift=5):
    """生成对应目标数据（保持相同批量）"""
    targets = np.zeros((sequence.shape[0], *sequence.shape[2:-1]))  # (batch, H, W)
    for b in range(sequence.shape[0]):
        last_frame = sequence[b, -1, ..., 0]
        targets[b] = np.roll(last_frame, shift=shift, axis=0)
    return targets[..., np.newaxis]  # 添加通道维度 (batch, H, W, 1)

def build_conv_lstm(input_shape=(5, 64, 64, 1)):
    model = Sequential([
        ConvLSTM2D(16, (3,3), padding='same', return_sequences=False, input_shape=input_shape),
        BatchNormalization(),
        Conv2D(1, 3, padding='same', activation='linear')
    ])
    return model

# 修改后的训练循环示例
class VisualizationCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 == 0:
            test_seq = generate_moving_square(batch_size=1)
            try:
                pred = self.model.predict(test_seq)
                # 确保预测结果维度正确
                if pred.ndim == 4:  # (batch, H, W, C)
                    target = np.roll(test_seq[0, -1, ..., 0], shift=5, axis=0)
                    target = target[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)
                else:
                    raise ValueError(f"Invalid prediction shape: {pred.shape}")
                
                plot_prediction(test_seq, target, pred, epoch)
            except Exception as e:
                print(f"Visualization error: {str(e)}")

# 修改模型初始化部分
def build_and_compile_model(input_shape=(5, 64, 64, 1)):
    """构建并编译模型"""
    model = build_conv_lstm(input_shape)
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['mae']
    )
    return model

# 修改训练部分
if __name__ == "__main__":
    # 生成匹配的数据对
    batch_size = 4  # 使用更大的批量
    test_sequence = generate_moving_square(batch_size=batch_size)
    target_frame = generate_target(test_sequence)  # 形状 (4, 64, 64, 1)
    
    # 验证数据形状
    print("输入形状:", test_sequence.shape)  # 应为 (4, 5, 64, 64, 1)
    print("目标形状:", target_frame.shape)   # 应为 (4, 64, 64, 1)
    
    # 初始化模型
    model = build_and_compile_model()
    
    # 添加回调
    vis_callback = VisualizationCallback()
    
    # 训练
    history = model.fit(
        test_sequence,
        target_frame,
        epochs=10,
        batch_size=2,  # 可小于总批量
        validation_split=0.2,
        callbacks=[vis_callback],
        verbose=1
    )
    
    # 绘制训练曲线
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # 最终预测可视化
    final_pred = model.predict(test_sequence)
    plot_prediction(test_sequence, target_frame, final_pred) 