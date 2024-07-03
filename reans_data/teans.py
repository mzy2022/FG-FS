from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data, meta = arff.loadarff('Openml_658.arff')
data = pd.DataFrame(data)

# data = pd.read_csv("Birds.csv")
# apd = pd.DataFrame(index=range(1483), columns=range(9))
# for a in range(0,1483):
#     each = data.iloc[a,:]
#     each_str = ' '.join(each)
#     float_numbers = []
#     words = each_str.split()
#     words.pop(0)
#     for num,word in enumerate(words):
#         if num != 8:
#             float_numbers.append(float(word))
#         else:
#             float_numbers.append(word)
#
#     apd.iloc[a,:] = float_numbers
#
# # print()
#
#
#
# # # 读取ARFF文件
#
#
df = data
encoder = LabelEncoder()

# 将数据转换为numpy数组
# array_data = np.array(data)
# df = pd.DataFrame(array_data)
new_df = pd.DataFrame()
dis = 1
k = 1
for num,name in enumerate(df.columns):
    # if num in range(len(df.columns))[-19:]:
    #     encoded_data = encoder.fit_transform(df.loc[:,name])
    #     new_df[f'Y{dis}'] = encoded_data
    #     dis += 1
    if name in ['SOUTH','SEX','UNION','RACE','OCCUPATION','SECTOR','MARR']:
        encoded_data = encoder.fit_transform(df.loc[:, name])
        new_df[name] = encoded_data
        k += 1
    else:
        new_df[name] = df.loc[:,name]
        k += 1
# 如果需要，可以进一步转换为pandas DataFrame
old_column_name = new_df.columns[-1]
#
# # 设置新列名

new_column_name = 'label'
# # 重命名最后一列
new_df.rename(columns={old_column_name: new_column_name}, inplace=True)

# columns = new_df.columns.tolist()

# 将倒数第六列与最后一列的索引位置交换
# columns[-6], columns[-1] = columns[-1], columns[-6]
# new_df = new_df[columns]
new_df.to_csv('Openml_658.csv',index=False)


# print(new_df)