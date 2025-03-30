import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(current_dir, 'mb_ulm_4con.csv')

try:
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    print(f"成功读取文件: {csv_file}")
    
    # 验证数据列
    required_columns = ['X', 'Y', 'ImageIndex']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV文件缺少必要列，需要包含X, Y, ImageIndex列")

    # 计算数据范围
    x_min, x_max = df['X'].min(), df['X'].max()
    y_min, y_max = df['Y'].min(), df['Y'].max()
    x_span = x_max - x_min
    y_span = y_max - y_min
    
    # 修正坐标范围计算
    x_min, x_max = df['X'].min() - 0.1*x_span, df['X'].max() + 0.1*x_span
    y_min, y_max = df['Y'].min() - 0.1*y_span, df['Y'].max() + 0.1*y_span
    
    # 根据数据范围动态计算图像尺寸（单位：英寸）
    dpi = 100
    fig_width = (y_span * 10) / dpi  # 交换XY跨度
    fig_height = (x_span * 10) / dpi
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_subplot(111)
    
    # 修改散点初始化部分
    scat = ax.scatter([], [], 
                     s=1, 
                     cmap='Reds',  # 使用红色渐变
                     edgecolors='none',
                     norm=plt.Normalize(vmin=df['ImageIndex'].min(), 
                                      vmax=df['ImageIndex'].max()))
    
    # 配置动画参数
    frame_step = max(1, len(df['ImageIndex'].unique()) // 100)  # 自动计算帧步长
    frames = sorted(df['ImageIndex'].unique()[::frame_step])
    
    # 初始化画布时设置固定坐标范围
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # 修改动画更新函数
    def update(frame):
        frame_data = df[df['ImageIndex'] <= frame]
        
        # 设置坐标和颜色
        scat.set_offsets(frame_data[['X', 'Y']].values)
        scat.set_array(frame_data['ImageIndex'])  # 用ImageIndex驱动颜色
        
        # 动态调整颜色范围
        scat.norm.vmax = frame  # 最新帧颜色最红
                # ... 其余轨迹绘制代码 ...
        
        return (scat,) + tuple(traj_lines.values())
    
    # 创建并保存动画
    ani = animation.FuncAnimation(
        fig, update, 
        frames=frames,
        interval=50, 
        blit=True
    )
    
    # 修改保存为动画
    output_file = os.path.join(current_dir, 'ulm_animation.gif')
    ani.save(output_file, writer='pillow', dpi=150)
    print(f"动画已保存至: {output_file}")

except FileNotFoundError:
    print(f"错误：文件 {csv_file} 未找到")
except pd.errors.EmptyDataError:
    print("错误：CSV文件为空或格式不正确")
except Exception as e:
    print(f"处理过程中发生错误: {str(e)}")
