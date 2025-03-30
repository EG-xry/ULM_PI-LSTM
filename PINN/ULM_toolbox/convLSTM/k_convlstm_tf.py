import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D

# 定义ConvLSTM模型构建函数
def build_conv_lstm(input_shape=(5, 64, 64, 1)):
    """
    构建ConvLSTM时序预测模型
    参数：
        input_shape : 输入张量形状 (时间步, 高度, 宽度, 通道数)
    返回：
        Keras Sequential模型
    """
    model = Sequential([
        # 第一层：ConvLSTM2D
        ConvLSTM2D(
            filters=16,              # 卷积滤波器数量，对应LSTM的隐藏单元数
            kernel_size=(3, 3),      # 卷积核尺寸
            padding='same',          # 保持输出尺寸与输入相同
            return_sequences=False,  # 只返回最后一个时间步的输出
            input_shape=input_shape, # 输入形状（不含批量维度）
            data_format='channels_last'  # 数据格式：批次最后
        ),
        # 批标准化层：加速训练，稳定收敛
        BatchNormalization(),
        # 输出卷积层：将特征映射到目标空间
        Conv2D(
            filters=1,               # 输出通道数（单帧预测）
            kernel_size=3,           # 卷积核尺寸
            padding='same',          # 保持输出尺寸
            activation='linear'      # 线性激活用于回归任务
        )
    ])
    return model

# === 数据准备部分 ===
# 定义模拟数据参数
batch_size = 2       # 批量大小
seq_length = 5       # 时间序列长度（过去帧数）
height, width = 64, 64  # 空间分辨率
channels = 1         # 输入通道数（灰度图）

# 生成随机输入数据（正态分布）
# 形状：(批量, 时间, 高, 宽, 通道)
inputs = tf.random.normal([batch_size, seq_length, height, width, channels])

# === 模型初始化 ===
model = build_conv_lstm()

# === 模型编译 ===
model.compile(
    loss='mse',  # 均方误差损失函数（回归任务常用）
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)  # Adam优化器
)

# 打印模型结构摘要
model.summary()

# === 前向传播测试 ===
outputs = model.predict(inputs)
print("\n输入形状:", inputs.shape)  # 应输出 (2, 5, 64, 64, 1)
print("输出形状:", outputs.shape)  # 应输出 (2, 64, 64, 1)

# === 训练示例 ===
# 生成随机目标数据（与输出形状一致）
targets = tf.random.normal([batch_size, height, width, 1])

# 执行模型训练
history = model.fit(
    inputs,
    targets,
    epochs=20,  # 训练轮次（用户实验发现需要较多轮次收敛）从2.118降到0.83
    verbose=1    # 显示训练进度
) 