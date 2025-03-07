import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D

class BubbleTracker:
    def __init__(self, input_shape=(5, 128, 128, 1)):
        self.model = self.build_model(input_shape)
        self.seq_length = input_shape[0]
    
    def build_model(self, input_shape):
        """构建ConvLSTM预测模型"""
        model = Sequential([
            ConvLSTM2D(filters=64, kernel_size=(3,3),
                      padding='same', return_sequences=True,
                      input_shape=input_shape),
            BatchNormalization(),
            ConvLSTM2D(filters=64, kernel_size=(3,3),
                      padding='same', return_sequences=False),
            BatchNormalization(),
            Conv3D(filters=1, kernel_size=(3,3,3),
                  padding='same', activation='sigmoid')
        ])
        model.compile(loss='mse', optimizer='adam')
        return model
    
    def create_sequences(self, heatmaps):
        """将热力图转换为训练序列"""
        sequences = []
        for i in range(len(heatmaps)-self.seq_length):
            seq = heatmaps[i:i+self.seq_length]
            target = heatmaps[i+self.seq_length]
            sequences.append((seq, target))
        return np.array(sequences)
    
    def train(self, coords_list, epochs=50):
        """
        训练模型
        :param coords_list: 微泡坐标序列 [[(x1,y1), (x2,y2), ...], ...]
        :param epochs: 训练轮次
        """
        # 生成热力图数据集
        heatmaps = self.generate_heatmaps(coords_list)
        
        # 创建训练数据
        sequences = self.create_sequences(heatmaps)
        X = np.array([s[0] for s in sequences])
        y = np.array([s[1] for s in sequences])
        
        # 训练模型
        self.model.fit(X, y, epochs=epochs, validation_split=0.2)
    
    def predict(self, past_coords):
        """
        预测下一时刻位置
        :param past_coords: 过去seq_length个坐标 [(x1,y1), ..., (xn,yn)]
        :return: 预测坐标(x,y)
        """
        # 生成输入热力图序列
        input_seq = self.generate_heatmaps([past_coords])[0]
        
        # 进行预测
        pred_heatmap = self.model.predict(np.array([input_seq]))[0,...,0]
        
        # 从热力图中提取最大响应位置
        y, x = np.unravel_index(np.argmax(pred_heatmap), pred_heatmap.shape)
        return (x, y)
    
    @staticmethod
    def generate_heatmaps(coords_list, img_size=(128,128), sigma=2):
        """
        生成高斯热力图
        :param coords_list: 坐标列表
        :param img_size: 图像尺寸
        :param sigma: 高斯核半径
        :return: 热力图序列 [num_frames, height, width, 1]
        """
        heatmaps = []
        for coords in coords_list:
            heatmap = np.zeros(img_size)
            for x, y in coords:
                xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
                dist = np.sqrt((xx-x)**2 + (yy-y)**2)
                heatmap += np.exp(-dist**2/(2*sigma**2))
            heatmaps.append(heatmap[..., np.newaxis])  # 添加通道维度
        return np.array(heatmaps)

# 使用示例
if __name__ == "__main__":
    # 生成模拟数据：直线运动的微泡
    num_samples = 100
    coords_data = [[(i, 64)] for i in range(64-20, 64+20)]  # 横向移动
    
    # 初始化跟踪器
    tracker = BubbleTracker(input_shape=(5, 128, 128, 1))
    
    # 训练模型
    tracker.train(coords_data, epochs=30)
    
    # 进行预测
    test_sequence = [(60,64), (61,64), (62,64), (63,64), (64,64)]
    predicted_pos = tracker.predict(test_sequence)
    print(f"预测位置: {predicted_pos} 实际位置: (65,64)") 