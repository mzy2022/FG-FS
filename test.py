import numpy as np

result = np.array([])  # 初始化为一个空数组
f_new = np.array([[1,2,3],[4,5,6],[7,8,9]])
for i in range(len(f_new)):
    result = np.concatenate((result, f_new[i]), axis=0)
print(f_new.T)