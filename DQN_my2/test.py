import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_score

data = pd.read_csv('data/AP_Omentum_Ovary.csv')
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
# selector = SelectKBest(mutual_info_classif, k=100).fit(X, y)
# cols = selector.get_support()
# X_new = X.loc[:, cols]
# new_df = pd.DataFrame()
# for num,col in enumerate(X_new.columns):
#     new_df[f'V{num}'] = X_new.loc[:,col]
#
# new_df['label'] = y
# new_df.to_csv('AP.csv',index=False)

clf = RandomForestClassifier(n_estimators=10, random_state=0)
scores = cross_val_score(clf, X, y, scoring='f1_micro', cv=5).mean()
print(scores)

# print(new_df)
