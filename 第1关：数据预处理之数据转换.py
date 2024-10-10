import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

data_url = "/data/workspace/myshixun/step1/train.csv"
df = pd.read_csv(data_url)

imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imp.fit(df.iloc[:,5:6])
X = imp.transform(df.iloc[:,5:6])
####### Begin ########
# 数据转换
scaler = preprocessing.StandardScaler()
scaler = scaler.fit_transform(X)
####### End ########
# 输出转换后的前6列数据
print(scaler[:6])
