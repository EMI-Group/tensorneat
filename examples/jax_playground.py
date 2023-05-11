import numpy as np

# 输入
a = np.array([1, 2, 3, 4])
b = np.array([5, 6])

# 创建一个网格，其中包含所有可能的组合
aa, bb = np.meshgrid(a, b)
aa = aa.flatten()
bb = bb.flatten()
print(aa, bb)