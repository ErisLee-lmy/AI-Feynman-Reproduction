import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 定义谐振子参数
m = 1.0  # 质量 (kg)
k = 2.0  # 劲度系数 (N/m)
omega = np.sqrt(k / m)  # 固有角频率 (rad/s)

# 初始条件
x0 = 1.0  # 初始位置 (m)
p0 = 0.0  # 初始动量 (kg·m/s)
t_max = 10.0  # 模拟时间 (s)
N = 1e3

dt = (t_max)/N
# 时间点
t = np.arange(0, t_max, dt)

# 初始化位置和动量数组
x = np.zeros_like(t)
p = np.zeros_like(t)

# 设置初始值
x[0] = x0
p[0] = p0

# 数值积分（使用简单的欧拉方法）
for i in range(1, len(t)):
    # 动量的变化
    p[i] = p[i - 1] - k * x[i - 1] * dt
    # 位置的变化
    x[i] = x[i - 1] + (p[i] / m) * dt

# 保存相空间轨迹到 CSV 文件
data = pd.DataFrame({'t': t, 'x': x, 'p': p})
csv_file = 'harmonic_oscillator_trajectory.csv'
data.to_csv(csv_file, index=False)
print(f"相空间轨迹已保存到文件：{csv_file}")

# 可视化相空间轨迹
plt.figure(figsize=(8, 6))
plt.plot(x, p, label='Phase Space Trajectory', color='blue')
plt.title('Phase Space Trajectory of 1D Harmonic Oscillator')
plt.xlabel('Position (x)')
plt.ylabel('Momentum (p)')
plt.grid()
plt.legend()
plt.show()
