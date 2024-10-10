# 对泰坦尼克号的数据进行清洗
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer 

# 读取数据
df = pd.read_csv('/data/workspace/myshixun/step1/train.csv')

##### begin #####
# 查看每列中是否存在空值
missing_values = df.isnull().any()  # 返回每列是否有缺失值的布尔值
print(missing_values)

# 使用SimpleImputer取出缺失值所在列的数值
age = df['Age'].values.reshape(-1, 1)  

# 实例化，均值填充
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

# fit_transform一步完成调取结果
imp_mean = imp.fit_transform(age)  

# 填充好的数据传回到 df['Age'] 列
df['Age'] = imp_mean  

# 检验是否还有空值，为0即说明空值均已被填充
print(df['Age'].isnull().sum())  
##### end #####    

# 正态分布离群点检测
##### begin #####
# 计算均值
mean_age = df['Age'].mean()

# 计算标准差
std_age = df['Age'].std()

# 识别异常值
# 定义离群点的条件
lower_bound = mean_age - 3 * std_age
upper_bound = mean_age + 3 * std_age

# 找到离群点
outliers = df[(df['Age'] < lower_bound) | (df['Age'] > upper_bound)]

# 输出离群点的详细信息
print(outliers)
##### end #####
