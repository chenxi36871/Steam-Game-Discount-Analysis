import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
df_model3 = pd.read_csv('D:/pythonProject/model/model_data.csv', index_col=0, encoding='utf_8_sig')
df_raw_data = pd.read_csv('D:/pythonProject/cleaning/raw_data_0401.csv', index_col=0, encoding='utf_8_sig')

df_model3 = df_raw_data.drop(['first_discount_date', 'fifty_discount_date', 'score_30days', 'rating_sample_num_30days', 'linux',
                             'win', 'mac', 'gamename', 'gameid', 'format_release_date'], axis=1)
df_model3 = df_model3.drop(['owner'], axis=1)
outlier1 = df_model3[df_model3['fifty_discount_period'] == 'None']
outlier1_index = list(outlier1.index)
df_model3 = df_model3.drop(index=outlier1_index) # 删除没有打过半价的游戏，剩余18395个数据

traindata=pd.DataFrame(df_model3,dtype=np.float)
traindata.to_csv('D:/pythonProject/model/traindata.csv', encoding='utf_8_sig')
traindata=pd.read_csv('D:/pythonProject/model/traindata.csv', index_col=0, encoding='utf_8_sig')

# 连续变量
conti_var = traindata[['release_price', 'avg_forever', 'avg_2weeks', 'median_forever',
                       'median_2weeks', 'ccu', 'reviewsummary_score', 'reviewsummary_forever_score',
                       'score_forever', 'rating_sample_num_forever', 'language_cnt',
                       'tagsum', 'first_discount_period', 'fifty_discount_period']]

heatmap(data=conti_var, figsize=(16, 15))
plt.show()

a = conti_var.corr()


# 有线性关系的：reviewsummary_score, reviewsummary_forever_score, release_price, score_forever, first_discount_period

# 分类变量
import statsmodels.api as sm
from statsmodels.formula.api import ols # ols 为建立线性回归模型的统计学库
from statsmodels.stats.anova import anova_lm
traindata = traindata.rename(columns={'features_Steam 排行榜': 'features_steam_rank',
                                      'features_Steam 创意工坊': 'features_steam_workshop',
                                      'features_Steam 成就': 'features_steam_achievement'})

lm = ols('fifty_discount_period ~ C(language_en) + C(language_cn)+C(language_ru)+'\
         'C(language_spanish)+C(language_french)+C(genre_独立)+C(genre_动作)+C(genre_休闲)+C(genre_冒险)+'\
         'C(genre_模拟)+C(genre_策略)+C(features_steam_achievement)+C(features_steam_workshop)+C(features_应用内购买)+C(features_单人)+'\
         'C(features_steam_rank)+C(features_远程畅玩)+C(features_多人)+C(features_支持控制器)', data=traindata).fit()
anova_result = anova_lm(lm)
# 显著的有：language_spanish, language_french, genre_独立, genre_动作,
# genre_休闲, genre_模拟, features_steam_achievement, features_steam_workshop,
# features_单人, features_远程畅玩, features_多人, features_支持控制器

# 多元线性回归建模
from statsmodels.formula.api import ols

lm = ols('fifty_discount_period ~ reviewsummary_score + reviewsummary_forever_score + release_price +'\
         'score_forever +first_discount_period + language_spanish+ language_french+'\
         'genre_独立 +genre_动作 +genre_休闲 +genre_模拟 +features_steam_achievement'\
         ' +features_steam_workshop +features_单人 +features_远程畅玩 +features_多人'\
         ' + features_支持控制器', data=traindata).fit()
lm.summary()

# 逐步回归
import statsmodels.formula.api as smf
import pandas as pd


def forward_selected(data, response):
    """前向逐步回归算法，源代码来自https://planspace.org/20150423-forward_selection_with_statsmodels/
    使用Adjusted R-squared来评判新加的参数是否提高回归中的统计显著性
    Linear model designed by forward selection.
    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response
    response: string, name of response column in data
    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()

    return model

model = forward_selected(traindata, 'fifty_discount_period')
print(model.model.formula)
# fifty_discount_period ~ first_discount_period + reviewsummary_forever_score + release_price +
#                       first_month_discount + reviewsummary_score + language_ru + genre_动作 +
#                       features_支持控制器 + features_steam_workshop + genre_休闲 + features_应用内购买 +
#                       tagsum + language_french + features_steam_achievement + genre_模拟 +
#                       features_远程畅玩 + genre_冒险 + language_spanish + language_en +
#                       features_多人 + genre_策略
print(model.params)
print(model.rsquared_adj)

# 检测多重共线性
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

x = traindata.drop(['fifty_discount_period'], axis=1)
# 共剩下21个变量
x = traindata[['first_discount_period', 'reviewsummary_forever_score', 'release_price',
               'first_month_discount', 'reviewsummary_score', 'language_ru', 'genre_动作',
               'features_支持控制器', 'features_steam_workshop', 'genre_休闲', 'features_应用内购买',
               'tagsum', 'language_french', 'features_steam_achievement', 'genre_模拟',
               'features_远程畅玩', 'genre_冒险', 'language_spanish', 'language_en', 'features_多人', 'genre_策略']]
# 当VIF<10,说明不存在多重共线性；当10<=VIF<100,存在较强的多重共线性，当VIF>=100,存在严重多重共线性
vif = [variance_inflation_factor(x.values, x.columns.get_loc(i)) for i in x.columns]
# language_en的VIF>10，存在较强的多重共线性

# PCA 主成分分析
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np

data = scale(x.values) # 标准化，标准化之后就自动根据协方差矩阵进行主成分分析了
# data2 = np.corrcoef(np.transpose(data)) # 没有必要单独计算协方差阵或相关系数阵
pca = PCA(n_components=20) # 可以调整主成分个数，n_components = 1
pca.fit(data)
print(pca.explained_variance_) # 输出特征根
print(pca.explained_variance_ratio_) # 输出解释方差比
a = pca.components_
print(pca.components_) # 输出主成分

# 可视化PCA方差贡献率
pca_var = pca.explained_variance_ratio_
plt.bar(range(1, 22), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, 22), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()

cum = 0
for num in range(0, len(pca_var)):
    cum = cum + pca_var[num]
    if num == 10:
        print(cum)
        break

    if cum > 0.8:
        print(num)
        break
# 取前13个主成分
data_new = pca.transform(data)
data_new = data_new[:, 0:15]
data_new = pd.DataFrame(data_new)
data_new = data_new.drop(['fifty_discount_period'], axis=1)
temp = pd.DataFrame(traindata['fifty_discount_period'])
temp = temp.reset_index(drop=True)
data_new['fifty_discount_period'] = temp

data_new = data_new.rename(columns={0: 'pca1',
                                     1: 'pca2',
                                     2: 'pca3',
                                     3: 'pca4',
                                     4: 'pca5',
                                     5: 'pca6',
                                     6: 'pca7',
                                     7: 'pca8',
                                     8: 'pca9',
                                     9: 'pca10',
                                    10: 'pca11',
                                    11: 'pca12',
                                    12: 'pca13',
                                    13: 'pca14',
                                    14: 'pca15'})

lm = ols('fifty_discount_period ~ pca1 + pca2 + pca3 +'\
         'pca4 +pca5 + pca6+ pca7+'\
         'pca8 +pca9 +pca10+pca11+pca12+pca13+pca14+pca15', data=data_new).fit()
lm.summary()


# 嵌套模型的似然比检验
model1 = ols('fifty_discount_period ~ reviewsummary_score + reviewsummary_forever_score + release_price +'\
         'score_forever +first_discount_period + language_spanish+ language_french+'\
         'genre_独立 +genre_动作 +genre_休闲 +genre_模拟 +features_steam_achievement'\
         ' +features_steam_workshop +features_单人 +features_远程畅玩 + features_支持控制器+ features_多人+ '\
         'genre_策略 + language_en', data=traindata).fit()
model1.summary()

model2 = ols('fifty_discount_period ~ reviewsummary_score + reviewsummary_forever_score + release_price +'\
         'score_forever +first_discount_period + language_spanish+ language_french+'\
         'genre_独立 +genre_动作 +genre_休闲 +genre_模拟 +features_steam_achievement'\
         ' +features_steam_workshop +features_单人 +features_远程畅玩 + features_支持控制器', data=traindata).fit()
model2.summary()


import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

anovaResults = anova_lm(model, model2)
print(anovaResults)

# 线性模型的诊断
import seaborn as sns


