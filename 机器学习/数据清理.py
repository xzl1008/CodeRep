# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 13:43:24 2018

@author: xiezhilong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#自定义frame
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'], 'year': [2000, 2001, 2002, 2001, 2002], 'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = pd.DataFrame(data)
print(frame.head())

csv_path = "housing.csv"
#导入外部的csv文件
housing = pd.read_csv(csv_path)
#显示导入的Dataframe的前面5行
print(housing.head())
#显示导入的Dataframe的倒数10行
print(housing.tail(10))
#显示Dataframe中的数据列，以及数据类型
print(housing.info())
#显示数据类型
type(housing)

#对于ocean_proximity这一列的值进行计数
housing["ocean_proximity"].value_counts()
#对于ocean_proximity这一列的值计数在图形中利用直方图显示
housing["ocean_proximity"].value_counts().plot(kind='barh')
plt.show()

#常用统计函数

#describe 针对Series或个DataFrame列计算汇总统计
#对于每一列进行统计分析，计数，平均值，标准差，最小值，四分位数的25%，50%，75%的值，最大值
housing.describe()

#count 非na值的数量
housing.count()
#min、max 计算最小值和最大值
housing.min()
housing.max()
#idxmin、idxmax 计算能够获取到最大值和最小值得索引值
housing.latitude.idxmin()
housing.longitude.idxmax()
#quantile 计算样本的分位数（0到1）,25%, 50%, 75%
housing.quantile([.25, .5, .75])
#sum 值的总和
housing.sum()
#mean 值得平均数
housing.mean()
#median 值得算术中位数（50%分位数）
housing.median()
#mad 根据平均值计算平均绝对离差
housing.mad()
#var 样本值的方差
housing.var()
#std 样本值的标准差
housing.std()
#skew 样本值得偏度（三阶矩）
housing.housing_median_age.skew()
#kurt 样本值得峰度（四阶矩）
housing.kurt()
#cumsum 样本值得累计和
housing.cumsum()
#cummin，cummax 样本值得累计最大值和累计最小值
housing.cummin()
housing.cummax()
#cumprod 样本值得累计积
housing.cumprod
#diff 计算一阶差分
housing.diff()
#pct_change 计算百分数变化
housing.housing_median_age.pct_change()

#用直方图的方式，显示所有列的分布情况，便于查看数据
#比如households，大于2000的有，但是数量很少，在清理时，需要进行处理，比如把所有大于2000的都作为2000的来处理
housing.hist(bins=50, figsize=(20,15))
plt.show()
#只显示一列的直方图
housing.households.hist(bins=50)
plt.show()
#将图片保留到文件
plt.savefig('filename.jpg')


#对数据进行拆分
from sklearn.model_selection import train_test_split
#拆分为训练数据和测试数据，测试数据占20%，random_state=42是为了保证每次取的都是相同的训练数据和测试数据，这样可以用于多个模型的比较
#因为如果模型使用不同的数据集进行训练和测试，效果比较没有太大意义，一般情况下都是设置为42
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# test_set.head()
train_set.shape
test_set.shape
housing.shape

housing["median_income"].hist()

housing["median_income"].head()
housing["median_income"].plot(kind='hist')
# Divide by 1.5 to limit the number of income categories,限制分类数量，除以1.5以后，可以减少分类数量
# 类似于分桶操作
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].plot(kind='hist')
# Label those above 5 as 5
#对新的列income_cat，对于标签值大于5的，按照5来处理
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
# housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=False)
housing["income_cat"].plot(kind='hist')

#分层随机抽样，对于训练目标，为了保证每个类别都可以取到数据进行训练（不然对于样本数量较多的类别，模型会有偏好），需要在抽样时，根据不同类别和比例来拆分训练和测试数据集
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# train_set, test_set = train_test_split(housing, stratify=housing['income_cat'])
    
housing["income_cat"].value_counts() / len(housing)


#对于收入类别（分为了5类），在整个数据集，分层抽样和随机抽样的不同方式下的占比以及误差进行比较
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
compare_props


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


#根据经度和维度显示散点图    
housing.plot(kind="scatter", x="longitude", y="latitude")
plt.show()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()

#
housing.plot(kind="scatter",  #图形类型，scatter为散点图，默认是折线图
             x="longitude",   #x轴
             y="latitude",    #y轴
             alpha=0.4,       #透明度，值越高，越不透明
             s=housing["population"]/100,  #圆点的大小
             label="population",   #标签名字
             figsize=(10,7),   #图形大小
             c="median_house_value",   #颜色，median_house_value的值越大，颜色越深
             cmap=plt.get_cmap("jet"),   #颜色映射表
             colorbar=True,  #图形旁边的颜色条
             sharex=False) 
#显示图例
plt.legend()
plt.show()
