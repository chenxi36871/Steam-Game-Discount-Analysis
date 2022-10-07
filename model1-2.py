import pandas as pd



df_model2 = pd.read_csv('D:/pythonProject/cleaning/raw_data_0401.csv', index_col=0, encoding='utf-8')
df_history_full_price = pd.read_csv('D:/pythonProject/cleaning/full_price_clean.csv', index_col=0, encoding='utf-8')

# 随机森林2
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 建立随机森林
rfc = RandomForestClassifier(n_estimators=100, random_state=90)

# 用交叉验证计算得分
score_pre = cross_val_score(rfc, x_train, y_train, cv=10).mean()
score_pre

# 调参，绘制学习曲线来调参n_estimators（对随机森林影响最大）
score_lt = []

# 每隔10步建立一个随机森林，获得不同n_estimators的得分
for i in range(0,100,10):
    rfc = RandomForestClassifier(n_estimators=i+1
                                ,random_state=90)
    score = cross_val_score(rfc, x_train, y_train, cv=10).mean()
    score_lt.append(score)
score_max = max(score_lt)
print('最大得分：{}'.format(score_max),
      '子树数量为：{}'.format(score_lt.index(score_max)*10+1))
# 最大得分：0.6735952586195937 子树数量为：191

# 绘制学习曲线
x = np.arange(1,101,10)
plt.subplot(111)
plt.plot(x, score_lt, 'o-')
plt.show()

# 在81附近缩小n_estimators的范围为30-49
score_lt = []
for i in range(70,89):
    rfc = RandomForestClassifier(n_estimators=i
                                ,random_state=90)
    score = cross_val_score(rfc, x_train, y_train, cv=10).mean()
    score_lt.append(score)
score_max = max(score_lt)
print('最大得分：{}'.format(score_max),
      '子树数量为：{}'.format(score_lt.index(score_max)+70))

# 绘制学习曲线
x = np.arange(70,89)
plt.subplot(111)
plt.plot(x, score_lt,'o-')
plt.show()

# 建立n_estimators为45的随机森林
rfc = RandomForestClassifier(n_estimators=81)

# 用网格搜索调整max_depth
param_grid = {'max_depth':np.arange(1,20)}
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(x_train, y_train)

best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)

# 用网格搜索调整max_features
param_grid = {'max_features':np.arange(6,43)}

rfc = RandomForestClassifier(n_estimators=81
                            # ,random_state=90
                            ,max_depth=18)
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(x_train, y_train)
best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)

# rfc classifier
rfc = RandomForestClassifier(n_estimators=81,
                             max_depth=18,
                             max_features=14)
score = cross_val_score(rfc, x_train, y_train, cv=10).mean()
rfc.fit(x_train, y_train)
importance = rfc.feature_importances_
indices = np.argsort(importance)[::-1]
features = x_train.columns
for f in range(x_train.shape[1]):
 print(("%2d) %-*s %f" % (f + 1, 30, features[f], importance[indices[f]])))

# 创建图
plt.figure()
plt.title("feature importance")
# features.shape[1]  数组的长度
plt.bar(range(features.shape[0]), importance[indices])
plt.xticks(range(features.shape[0]), features, rotation=90)
plt.subplots_adjust(bottom=0.5)
plt.show()

# ROC plot

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', linewidth=lw, label='ROC curve (area = %0.4f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', linewidth=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()





