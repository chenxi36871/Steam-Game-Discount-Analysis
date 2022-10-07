import pandas as pd

df_model = pd.read_csv('D:/pythonProject/model/model_data.csv', index_col=0, encoding='utf_8_sig')
df_history_price = pd.read_csv('D:/pythonProject/cleaning/history_price_clean.csv', index_col=0, encoding='utf-8')
df_release_price = pd.read_csv('D:/pythonProject/cleaning/release_price_clean.csv', index_col=0, encoding='utf-8')


# 获得第一次打折的价格
first_discount = pd.DataFrame(columns=('gameid', 'first_discount_price'))
gameid_list = list(df_history_price)
for x in range(0, df_history_price.shape[1]): # df_history_price.shape[1]
    real_num = len(df_history_price) - df_history_price.iloc[:, x].isnull().sum() - 1
    r_price = df_release_price.iloc[x, 1]
    while real_num >= 0:
        if df_history_price.iloc[real_num, x] < r_price:
            first_discount_price = df_history_price.iloc[real_num, x]
            first_discount = first_discount.append([{'gameid': int(gameid_list[x]),
                                                     'first_discount_price': first_discount_price}],
                                                   ignore_index=True)
            break
        if real_num == 0:
            first_discount = first_discount.append([{'gameid': int(gameid_list[x]),
                                                     'first_discount_price': 'None'}],
                                                   ignore_index=True)
        real_num = real_num - 1

df_release_price.isnull().sum()
# 有4206个game没有价格信息，剩余35620个。None表示没有打折过
df_release_price = df_release_price.dropna(axis=0, how='any')

df_price = pd.merge(df_release_price, first_discount, how='left', on='gameid')

a1 = pd.DataFrame(columns=('gameid', 'first_discount_percent'))
for num in range(0, len(df_price)):
    if df_price.iloc[num, 2] == 'None':
        a1 = a1.append([{"gameid": df_price.iloc[num, 0],
                         'first_discount_percent': 'None'}],
                       ignore_index=True)
        continue
    diff = df_price.iloc[num, 1] - df_price.iloc[num, 2]
    perc = round(diff/df_price.iloc[num, 1], 3)
    a1 = a1.append([{"gameid": df_price.iloc[num, 0],
                     'first_discount_percent': perc}],
                   ignore_index=True)

df_price = pd.merge(df_price, a1, how='left', on='gameid')
df_price.to_csv('D:/pythonProject/model/price_discount.csv', encoding='utf_8_sig')

##################################################################
df_model = pd.read_csv('D:/pythonProject/model/model_data.csv', index_col=0, encoding='utf_8_sig')
df_price = pd.read_csv('D:/pythonProject/model/price_discount.csv', index_col=0, encoding='utf_8_sig')
df_raw_data = pd.read_csv('D:/pythonProject/cleaning/raw_data_final.csv', index_col=0, encoding='utf_8_sig')

a = df_raw_data['gameid']
df_price = df_price.rename(columns={'gameid': 'index'})
df_price['gameid'] = a
a = pd.concat([df_raw_data, df_price], axis=1)
a = a.drop(['description'], axis=1)

df = a.dropna(axis=0, how='any') # 删除248个剩余35372个
df = df.reset_index(drop=True)
a1= df.iloc[:,[1,45,46,47]]
a1 = a1[a1['release_price'] <= 200]

a = list(a1['first_discount_price'])
b = list(a1['first_discount_percent'])
df_model['first_discount_price'] = a
df_model['first_discount_percent'] = b

df_model.to_csv('D:/pythonProject/model/model_data2.csv', encoding='utf_8_sig')
#################################################回归
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
df_model2 = pd.read_csv('D:/pythonProject/model/model_data2.csv', index_col=0, encoding='utf_8_sig')

df_model2 = df_model2.drop(['first_discount_price', 'first_month_discount'], axis=1)
x = df_model2[df_model2['first_discount_percent'] == 'None']
null_index = list(x.index)
df_model2 = df_model2.drop(index=null_index) # 删除没有打折过的数据，还剩下29673条
df_model2 = df_model2.reset_index(drop=True)

df_model2 = df_model2.rename(columns={'features_Steam 创意工坊': 'features_steam_workshop'})

# 连续变量：release_price, avg_forever, avg_2weeks, median_forever, median_2weeks, ccu,
# reviewsummary_score, reviewsummary_forever_score, score_forever, rating_sample_num_forever,
# language_cnt, tagsum, system_cnt
conti_var = df_model2[['release_price', 'avg_forever', 'avg_2weeks', 'median_forever',
                       'median_2weeks', 'ccu', 'reviewsummary_score', 'reviewsummary_forever_score',
                       'score_forever', 'rating_sample_num_forever', 'language_cnt',
                       'tagsum', 'system_cnt']]
y = df_model2['first_discount_percent']
for num in range(0, len(df_model2)):
    df_model2.loc[num, 'first_discount_percent'] = float(df_model2.loc[num, 'first_discount_percent'])

person_var = pd.concat([conti_var, y], axis=1)
person_var[['first_discount_percent']] = person_var[['first_discount_percent']].astype(float)
x = person_var.corr()
conti_var_list = list(person_var)
# release_price, reviewsummary_forever_score, score_forever, tagsum
for num in range(0,13):
    r, p = stats.pearsonr(person_var.iloc[:, num], y)
    print(conti_var_list[num])
    print('r=' ,r)
    print('p=', p)

# 热力图
import seaborn as sns
def heatmap(data, method='pearson', camp='RdYlGn', figsize=(10 ,8)):
    """
    data: 整份数据
    method：默认为 pearson 系数
    camp：默认为：RdYlGn-红黄蓝；YlGnBu-黄绿蓝；Blues/Greens 也是不错的选择
    figsize: 默认为 10，8
    """
    ## 消除斜对角颜色重复的色块
    #     mask = np.zeros_like(df2.corr())
    #     mask[np.tril_indices_from(mask)] = True
    plt.figure(figsize=figsize, dpi= 80)
    sns.heatmap(data.corr(method=method), \
                xticklabels=data.corr(method=method).columns, \
                yticklabels=data.corr(method=method).columns, cmap=camp, \
                center=0, annot=True)
    # 要想实现只是留下对角线一半的效果，括号内的参数可以加上 mask=mask

heatmap(data=person_var, figsize=(16,15))
plt.show()

# 分类变量
nominal_vars = list(x_train)
nominal_vars_list = []
for x in nominal_vars:
    if x not in conti_var_list:
        nominal_vars_list.append(x)
for each in nominal_vars_list:
    print(each, ':')
    print(df_model2[each].agg(['value_counts']).T)
    print('='*35)  # 发现有一些分类变量分布极其不均衡

# 方差分析
import statsmodels.api as sm
from statsmodels.formula.api import ols # ols 为建立线性回归模型的统计学库
from statsmodels.stats.anova import anova_lm
df_model2 = df_model2.rename(columns={'features_Steam 排行榜': 'features_steam_rank'})

lm = ols('first_discount_percent ~ C(language_en) + C(language_cn)+C(language_ru)+'\
         'C(language_spanish)+C(language_french)+C(genre_独立)+C(genre_动作)+C(genre_休闲)+C(genre_冒险)+'\
         'C(genre_模拟)+C(genre_策略)+C(features_steam_achievement)+C(features_steam_workshop)+C(features_应用内购买)+C(features_单人)+'\
         'C(features_steam_rank)+C(features_远程畅玩)+C(features_多人)+C(features_支持控制器)', data=df_model2).fit()

anova_lm(lm)


# 拟合线性回归模型
# 划分训练集和测试集
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df_model2, test_size=0.2, random_state=42)

x_train = train_set.drop(['first_discount_percent'], axis=1)
x_test = test_set.drop(['first_discount_percent'], axis=1)
y_train = train_set['first_discount_percent']
y_test = test_set['first_discount_percent']

# model
from statsmodels.formula.api import ols

lm = ols('first_discount_percent ~ release_price + reviewsummary_forever_score + '
         'score_forever+tagsum+', data=df).fit()
lm.summary()


