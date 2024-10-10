import pandas as pd
from imblearn.over_sampling import RandomOverSampler

data_url = "/data/workspace/myshixun/step1/diabetes.csv"
df = pd.read_csv(data_url)
X = df.iloc[:,0:8]
y=df.iloc[:,8]
###### Begin ######
# 随机过采样
ros = RandomOverSampler(random_state=0)
###### End ######
X_resampled, y_resampled = ros.fit_resample(X, y)
# 显示采样后的数据
print(X_resampled.info)
