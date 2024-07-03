import pandas as pd

data = pd.read_csv('BMI.csv')
data1 = data.copy()
df = data1.drop(columns=['name'])
df.to_csv('BMI.csv',index=False)
