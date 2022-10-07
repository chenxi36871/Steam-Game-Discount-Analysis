import math

import pandas as pd
import missingno as msno
from matplotlib import pyplot as plt
from collections import Counter
from datetime import datetime

# df_raw_data_new.to_csv('D:/pythonProject/cleaning/raw_data_final.csv', encoding='utf_8_sig')
df_raw_data = pd.read_csv('D:/pythonProject/cleaning/raw_data_final.csv', index_col=0, encoding='utf_8_sig')
df_raw_data = pd.read_csv('D:/pythonProject/cleaning/raw_data_noout.csv', index_col=0, encoding='utf_8_sig')
df_raw_data['tagsum'] = tagsum
df_raw_data.to_csv('D:/pythonProject/cleaning/raw_data_0401.csv', encoding='utf_8_sig')
df_raw_data = pd.read_csv('D:/pythonProject/cleaning/raw_data_0401.csv', index_col=0, encoding='utf_8_sig')
df = df_raw_data.drop(['description'], axis=1)

# 字符串'yyyy-mm-dd'转datetime格式
def str_to_date(text):
    temp = text.split('-')
    year = int(temp[0])
    month = int(temp[1])
    day = int(temp[2])
    d = datetime(year, month, day)
    return d

for num in range(0, len(df_raw_data)):
    df_raw_data.loc[num, 'format_release_date'] = str_to_date(df_raw_data.loc[num, 'format_release_date'])

# visualize missing value
msno.matrix(df_raw_data, labels=True)
plt.show()
msno.bar(df_raw_data)
plt.show() # owner, avg_forever, avg_2weeks, median_~, ccu
df['ccu'].isnull().sum() # 248个缺失值
a = pd.DataFrame([df['ccu'].isnull()]).T
a = list(a[a['ccu'] == True].T)
df = df.drop(a, axis=0)
df = df.dropna(axis=0, how='any') # 删除248个剩余35372个

#  missingno相关性热图措施无效的相关性：一个变量的存在或不存在如何强烈影响的另一个的存在：
msno.heatmap(df_raw_data)
plt.show()

# 异常值处理
plt.boxplot(df_raw_data['release_price'])
plt.show()

outlier1 = df_raw_data[df_raw_data['release_price'] > 200]
outlier1_index = list(outlier1.index)
df_raw_data = df_raw_data.drop(index=outlier1_index)
df_raw_data.to_csv('D:/pythonProject/cleaning/raw_data_noout.csv', encoding='utf_8_sig')

out = df_raw_data[df_raw_data['release_price'] <= 200]

plt.hist(df_raw_data['release_price'], bins=30)
plt.show()
plt.boxplot(out2["avg_forever"])
plt.show()
out2 = df_raw_data[df_raw_data['avg_forever'] < 400]

# 可视化
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False
import seaborn as sns
sns.set(style="white") #设置seaborn画图的背景为白色
sns.set(style="whitegrid", color_codes=True)
# 游戏基本信息
# 价格：发行时间
plt.scatter(df_raw_data['format_release_date'], df_raw_data['release_price']) #You can also add more variables here to represent color and size.
plt.show()
plt.scatter(out['format_release_date'], out['release_price'], s=5)
plt.show()

# 价格：特征features -- 分组箱线图

# 价格：系统（分类or 支持数量）-- 分组箱线图
# 价格：类别 – 分组箱线图
# 价格：语言 – 分组箱线图

# 一个月是否打折：发行时间
# 一个月是否打折：特征
# 一个月是否打折：系统（分类or 支持数量）
df_raw_data = df_raw_data.reset_index(drop=True)
system_cnt = []
for num in range(0, len(df_raw_data)):
    system_cnt.append(df_raw_data.loc[num, 'win'] + df_raw_data.loc[num, 'mac'] + df_raw_data.loc[num, 'linux'])
df_raw_data['system_cnt'] = system_cnt
df_model['system_cnt'] = system_cnt

# 一个月是否打折：类别
# 一个月是否打折：语言cnt
plt.figure(figsize=(15,8))
ax = sns.kdeplot(df_raw_data["language_cnt"][df_raw_data.first_month_discount == 0], color="darkturquoise", shade=True)
sns.kdeplot(df_raw_data["language_cnt"][df_raw_data.first_month_discount == 1], color="lightcoral", shade=True)
plt.legend(['Not-discount', 'Discount'])
# plt.title('Density Plot of Age for Surviving Population and Deceased Population')
ax.set(xlabel='language_cnt')
plt.xlim(0,35)
plt.show()

# 一个月是否打折：用户自定义标签cnt
plt.figure(figsize=(15,8))
ax = sns.kdeplot(df_raw_data["tagsum"][df_raw_data.first_month_discount == 0], color="darkturquoise", shade=True)
sns.kdeplot(df_raw_data["tagsum"][df_raw_data.first_month_discount == 1], color="lightcoral", shade=True)
plt.legend(['Not-discount', 'Discount'])
ax.set(xlabel='tagsum')
plt.xlim(0,25)
plt.show()

# 玩家信息
# reviewsummary_score
plt.figure(figsize=(15,8))
ax = sns.kdeplot(df_raw_data["reviewsummary_score"][df_raw_data.first_month_discount == 0], color="darkturquoise", shade=True)
sns.kdeplot(df_raw_data["reviewsummary_score"][df_raw_data.first_month_discount == 1], color="lightcoral", shade=True)
plt.legend(['Not-discount', 'Discount'])
ax.set(xlabel='reviewsummary_score')
plt.xlim(-1,10)
plt.show()

plt.figure(figsize=(15,8))
ax = sns.kdeplot(df_raw_data["reviewsummary_forever_score"][df_raw_data.first_month_discount == 0], color="darkturquoise", shade=True)
sns.kdeplot(df_raw_data["reviewsummary_forever_score"][df_raw_data.first_month_discount == 1], color="lightcoral", shade=True)
plt.legend(['Not-discount', 'Discount'])
ax.set(xlabel='reviewsummary_forever_score')
plt.xlim(-1,10)
plt.show()

sns.boxplot(x=df_raw_data['first_month_discount'], y=df_raw_data['reviewsummary_forever_score'])
plt.show()

# score
plt.figure(figsize=(15,8))
ax = sns.kdeplot(df_raw_data["score_forever"][df_raw_data.first_month_discount == 1], color="darkturquoise", shade=True)
sns.kdeplot(df_raw_data["score_forever"][df_raw_data.first_month_discount == 0], color="lightcoral", shade=True)
plt.legend(['Not-discount', 'Discount'])
ax.set(xlabel='score_forever')
plt.xlim(0.25,1.2)
plt.show()

sns.boxplot(x=df_raw_data['first_month_discount'], y=df_raw_data['score_forever'])
plt.show()

# rating sample num
# rating_sample_num_30days, rating_sample_num_forever
plt.figure(figsize=(15,8))
ax = sns.kdeplot(df_raw_data["rating_sample_num_forever"][df_raw_data.first_month_discount == 0], color="darkturquoise", shade=True)
sns.kdeplot(df_raw_data["rating_sample_num_forever"][df_raw_data.first_month_discount == 1], color="lightcoral", shade=True)
plt.legend(['Not-discount', 'Discount'])
ax.set(xlabel='rating_sample_num_forever')
plt.xlim(-2000,4000)
plt.show()

sns.boxplot(x=df_raw_data['first_month_discount'], y=df_raw_data['rating_sample_num_forever'])
plt.show()

sns.distplot(df_raw_data["avg_forever"])
plt.show()

# avg playtime
# avg_forever, avg_2weeks
plt.figure(figsize=(15,8))
ax = sns.kdeplot(df_raw_data["avg_forever"][df_raw_data.first_month_discount == 0], color="darkturquoise", shade=True)
sns.kdeplot(df_raw_data["avg_forever"][df_raw_data.first_month_discount == 1], color="lightcoral", shade=True)
plt.legend(['Not-discount', 'Discount'])
ax.set(xlabel='avg_forever')
plt.xlim(-500,1000)
plt.show()



