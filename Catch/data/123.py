import pandas as pd


# print(data)
# label = data['label']
# del data['label']
# data['label'] = label
# print(data)
data = pd.read_hdf('PimaIndian.hdf')
print(data)
