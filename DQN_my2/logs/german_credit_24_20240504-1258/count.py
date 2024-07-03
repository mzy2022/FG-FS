import numpy as np
import pandas as pd

data_test = pd.read_csv('test.csv').iloc[:, :-1]

columns = data_test.columns.tolist()
ss = ["abss", 'square', 'inverse', 'log', 'sqrt', 'power3']
countlist = np.zeros(len(columns))

for num, column in enumerate(columns):
    a = column.count('_')
    bin_num = column.count('bin_')
    abss = column.count('abss')
    square = column.count('square')
    inverse = column.count('inverse')
    log = column.count('log')
    sqrt = column.count('sqrt')
    power3 = column.count('power3')

    countlist[num] += (a - abss - square - inverse - log - sqrt - power3 - bin_num) / 2 + abss + square + inverse + log + sqrt + power3

count_dict = {}
for list in countlist:
    list = int(list)
    if list in count_dict:
        count_dict[list] += 1
    else:
        count_dict[list] = 1

print(count_dict)
