import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import check_estimator
from autofeat import AutoFeatRegressor, AutoFeatClassifier
import matplotlib.pyplot as plt

np.random.seed(10)
x1 = np.random.rand(1000)
x2 = np.random.randn(1000)
x3 = np.random.rand(1000)
target = 2 + 15 * x1 + 3 / (x2 - 1 / x3) + 5 * (x2 + np.log(x1)) ** 3
target_noisy = target + 0.01 * target.std() * np.random.randn(1000)
target_very_noisy = target + 0.1 * target.std() * np.random.randn(1000)
X = np.vstack([x1, x2, x3]).T
df_org = pd.DataFrame(X, columns=["x1", "x2", "x3"])


# autofeat with different number of feature engineering steps
# 3 are perfect
for steps in range(5):
    np.random.seed(55)
    print("### AutoFeat with %i feateng_steps" % steps)
    afreg = AutoFeatRegressor(verbose=1, feateng_steps=steps)
    df = afreg.fit_transform(df_org, target)
    r2 = afreg.score(df_org, target)
    print("## Final R^2: %.4f" % r2)
    plt.figure()
    plt.scatter(afreg.predict(df_org), target, s=2)
    plt.title("%i FE steps (R^2: %.4f; %i new features)" % (steps, r2, len(afreg.new_feat_cols_)))

# afreg = AutoFeatRegressor(verbose=1, feateng_steps=3)
# # train on noisy data
# df = afreg.fit_transform(df_org, target_noisy)
# # test on real targets
# print("Final R^2: %.4f" % afreg.score(df, target))
# plt.figure()
# plt.scatter(afreg.predict(df), target, s=2)
plt.show()