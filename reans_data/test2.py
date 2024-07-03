import pandas as pd

df = pd.read_csv('liver_disorders.csv')
columns = df.columns.tolist()

# 交换最后两列
columns[-1], columns[-2] = columns[-2], columns[-1]

# 重新排列DataFrame的列顺序
df = df[columns]

df.to_csv('liver_disorders.csv',index=False)