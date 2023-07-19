import numpy as np


vals = np.array([1, 2])
weights = np.array([[0, 4], [5, 0]])

ins1 = vals * weights[:, 0]
ins2 = vals * weights[:, 1]
ins_all = vals * weights.T

print(ins1)
print(ins2)
print(ins_all)