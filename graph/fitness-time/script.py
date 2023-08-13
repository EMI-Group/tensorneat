from matplotlib import pyplot as plt
import numpy as np

# 使用 genfromtxt 函数读取 CSV 文件
data = np.genfromtxt('neatpython.csv', delimiter=',', skip_header=1)  # 假设有一个头部行
mean_time, fitness_mean = data.T

fig, ax = plt.subplots()
ax.plot(mean_time, fitness_mean, color='green', label='NEAT-Python', linestyle=':')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Average Fitness')
ax.set_xlim(0, 500)
ax.set_ylim(-2000, -1000)
ax.legend()


# ci = 1.96 * neatax_sem
# lower_bound = neatax_mean - ci
# upper_bound = neatax_mean + ci
# plt.plot(mean_time, fitness_mean, color='r', label='NEAT-Python')
# plt.fill_between(x_axis, lower_bound, upper_bound, color='red', alpha=0.2)
fig.show()
