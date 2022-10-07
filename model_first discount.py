import pandas as pd
df_model = df_raw_data.drop(['first_discount_date', 'fifty_discount_date', 'score_30days', 'rating_sample_num_30days', 'linux',
                             'win', 'mac', 'gamename', 'gameid', 'format_release_date',
                             'first_discount_period',
                             'fifty_discount_period'], axis=1)
df_model.to_csv('D:/pythonProject/model/model_data.csv', encoding='utf_8_sig')
owner_dummy = pd.get_dummies(df_model['owner'], drop_first=False, prefix='owner')
owner_dummy = owner_dummy.iloc[:, 0:11] #避免多重共线性
df_model = pd.concat([df_model, owner_dummy], axis=1)
df_model = df_model.drop(['owner'], axis=1)

import pandas as pd
df_model = pd.read_csv('D:/pythonProject/model/model_data.csv', index_col=0, encoding='utf_8_sig')
# 划分训练集和测试集
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df_model, test_size=0.2, random_state=42)

x_train = train_set.drop(['first_month_discount'], axis=1)
x_test = test_set.drop(['first_month_discount'], axis=1)
y_train = train_set['first_month_discount']
y_test = test_set['first_month_discount']
# 用逻辑回归判断是否会在一个月内打折
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


####################################################### 决策树model
import sklearn.tree as tree

# 直接使用交叉网格搜索来优化决策树模型，边训练边优化
from sklearn.model_selection import GridSearchCV
# 网格搜索的参数：正常决策树建模中的参数 - 评估指标，树的深度，
 ## 最小拆分的叶子样本数与树的深度
param_grid = {'criterion': ['entropy', 'gini'],
             'max_depth': [2, 3, 4, 5, 6, 7, 8],
             'min_samples_split': [4, 8, 12, 16, 20, 24, 28]}
                # 通常来说，十几层的树已经是比较深了

dtree = tree.DecisionTreeClassifier()  # 定义一棵树
dtree_cv = GridSearchCV(estimator=dtree, param_grid=param_grid,
                            scoring='roc_auc', cv=4)
dtree_cv.fit(X=x_train, y=y_train)
dtree_cv.best_params_
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=24)
dtree.fit(x_train, y_train)

# 使用模型来对测试集进行预测
test_est = dtree.predict(x_test)

# 模型评估
import sklearn.metrics as metrics

test_est = dtree.predict(x_test)
print('决策树精确度...')
print(metrics.classification_report(test_est, y_test))
print('决策树 AUC...')
fpr_test, tpr_test, th_test = metrics.roc_curve(test_est, y_test)
print('decision tree在训练集上的accuracy:', dtree.score(x_train,y_train))
print('decision tree在测试集上的accuracy:', dtree.score(x_test,y_test))
print('decision tree的AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

print("决策树准确度:")
print(metrics.classification_report(y_test,test_est))
        # 该矩阵表格其实作用不大
print("决策树 AUC:")
fpr_test, tpr_test, th_test = metrics.roc_curve(y_test, test_est)
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))
print(clfcv.best_params_)

# 可视化决策树

from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from sklearn import tree
fn = list(x_train)
cn=['No', 'Yes']
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=300)
tree.plot_tree(clf,
               feature_names=fn,
               class_names=cn,
               filled=True);
fig.savefig('D:/pythonProject/model/imagename.png')

##################################################################### 用随机森林进行预测
param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[5, 6, 7, 8],    # 深度：这里是森林中每棵决策树的深度
    'n_estimators':[11,13,15],  # 决策树个数-随机森林特有参数
    'max_features':[0.3,0.4,0.5],
     # 每棵决策树使用的变量占比-随机森林特有参数（结合原理）
    'min_samples_split':[4,8,12,16]  # 叶子的最小拆分样本量
}

import sklearn.ensemble as ensemble # ensemble learning: 集成学习

rfc = ensemble.RandomForestClassifier()
rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid,
                      scoring='roc_auc', cv=4)
rfc_cv.fit(x_train, y_train)
rfc_cv.best_params_

# 使用随机森林对测试集进行预测
test_est = rfc_cv.predict(x_test)
print('随机森林精确度...')
print(metrics.classification_report(test_est, y_test))
print('随机森林 AUC...')
fpr_test, tpr_test, th_test = metrics.roc_curve(test_est, y_test)
     # 构造 roc 曲线
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))
# random forest
rfc = ensemble.RandomForestClassifier(criterion='entropy', max_depth=8, max_features=0.3, min_samples_split=12,
                                      n_estimators=15)
rfc.fit(x_train,y_train)
test_est = rfc.predict(x_test)
print('随机森林精确度...')
print(metrics.classification_report(test_est, y_test))
print('随机森林 AUC...')
fpr_test, tpr_test, th_test = metrics.roc_curve(test_est, y_test)
     # 构造 roc 曲线
print('random forest的AUC = %.4f' %metrics.auc(fpr_test, tpr_test))  # AUC=0.7290
print('random forest在训练集上的accuracy:', rfc.score(x_train,y_train))
print('random forest在测试集上的accuracy:', rfc.score(x_test,y_test))

# roc curve
test_est = rfc.predict(x_train)
fpr_train, tpr_train, thresholds = metrics.roc_curve(test_est, y_train)
test_est = rfc.predict(x_test)
fpr_test, tpr_test, thresholds = metrics.roc_curve(test_est, y_test)

auc_value = metrics.auc(fpr_test, tpr_test)
plt.figure()
lw = 2
plt.plot(fpr_test, tpr_test, color='darkorange', linewidth=lw, label='ROC curve (area = %0.4f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', linewidth=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('random forest ROC curve (test)')
plt.legend(loc="lower right")
plt.show()


# 特征重要性
import numpy as np
y_train = np.ravel(y_train)
rfc.fit(x_train, y_train)
importance = rfc.feature_importances_
indices = np.argsort(importance)[::-1]
features = x_train.columns
for f in range(x_train.shape[1]):
 print(("%2d) %-*s %f" % (f + 1, 30, features[f], importance[indices[f]])))

data_features_part = list(x_train.columns)
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False

plt.figure()
plt.title("random forest feature importance")
sns.barplot(y=data_features_part, x=rfc.feature_importances_)
plt.subplots_adjust(left=0.4)
plt.show()

#########################################################################################################
# GBDT model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV

# 模型训练，使用GBDT算法
gbdt = GradientBoostingClassifier()
param_grid = {
    'min_sample_split':[10,20,30],
    'n_estimators':[30,50,70,90],
    'max_depth':[14,16,18],

}
xgbt_gr = GridSearchCV(xgbt, param_grid, scoring='roc_auc', cv=4)
gbr = GradientBoostingClassifier(n_estimators=100, max_depth=5, min_samples_split=3, learning_rate=0.1)
gbr.fit(x_train, y_train.ravel())
joblib.dump(gbr, 'D:/pythonProject/GBDT_train_model_result.m')   # 保存模型
# 训练和验证的准确率
y_gbr = gbr.predict(x_train)
y_gbr1 = gbr.predict(x_test)
acc_train = gbr.score(x_train, y_train)
acc_test = gbr.score(x_test, y_test)
print(acc_train) #0.7117846318416831
print(acc_test)  #0.666999429549344

# GBDT调参
# 不用任何参数的情况下
gbm0 = GradientBoostingClassifier(random_state=10)
gbm0.fit(x_train,y_train)
y_pred = gbm0.predict(x_test)
y_predprob = gbm0.predict_proba(x_test)[:,1]
print("Accuracy : %.4g" % metrics.accuracy_score(y_test.values, y_pred))
print("AUC Score (test): %f" % metrics.roc_auc_score(y_test, y_predprob))

# 将步长初始值设置为0.1。对于迭代次数进行网格搜索如下：
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10),
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(x_train,y_train)
print(gsearch1.best_params_) # best=70
print(gsearch1.best_score_)  # best score=0.7177378688092927

# 找到了一个合适的迭代次数，现在我们开始对决策树进行调参。首先我们对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索。
param_test2 = {'max_depth':range(14,18,1), 'min_samples_split':range(10,50,10)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70, min_samples_leaf=20,
      max_features='sqrt', random_state=10),
   param_grid = param_test2, scoring='roc_auc', cv=5)
gsearch2.fit(x_train,y_train)
print(gsearch2.best_params_) #best max_depth=16, best min_sample_split=10
print(gsearch2.best_score_)  #0.720960165968614
# 定下max_depth，对于内部节点再划分所需最小样本数min_samples_split，我们暂时不能一起定下来，因为这个还和决策树其他的参数存在关联。
# 下面我们再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参
param_test3 = {'min_samples_split':range(3,10,2), 'min_samples_leaf':range(10,70,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70,max_depth=16,
                                     max_features='sqrt', random_state=10),
                       param_grid = param_test3, scoring='roc_auc', cv=5)
gsearch3.fit(x_train,y_train)
print(gsearch3.best_params_) #{'min_samples_leaf': 10, 'min_samples_split': 3}
print(gsearch3.best_score_) # 0.7239756787992104

# 可以都放到GBDT类里面去看看效果了。现在我们用新参数拟合数据：
gbm1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70,max_depth=16, min_samples_leaf =10,
               min_samples_split =3, max_features='sqrt', random_state=10)
gbm1.fit(x_train,y_train)
y_pred = gbm1.predict(x_test)
y_predprob = gbm1.predict_proba(x_test)[:,1]
print ("GBDT在训练集上的accuracy : %.4g" % metrics.accuracy_score(y_train.values, gbm1.predict(x_train)))
print ("GBDT在测试集上的accuracy : %.4g" % metrics.accuracy_score(y_test.values, y_pred))
print ("GBDT的AUC Score: %f" % metrics.roc_auc_score(y_test, y_predprob)) #Accuracy : 0.6788, AUC Score (test): 0.722163

# 特征重要性
data_features_part = list(x_train.columns)
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False

plt.figure()
plt.title("GBDT feature importance")
sns.barplot(y=data_features_part, x=gbm1.feature_importances_)
plt.subplots_adjust(left=0.4)
plt.show()


#绘制ROC曲线
test_est = gbm1.predict(x_train)
fpr_train, tpr_train, thresholds = metrics.roc_curve(test_est, y_train)

test_est = gbm1.predict(x_test)
fpr_test, tpr_test, thresholds = metrics.roc_curve(test_est, y_test)

auc_value = metrics.auc(fpr_test, tpr_test)
auc_value = 0.7222
plt.figure()
lw = 2
plt.plot(fpr_test, tpr_test, color='darkorange', linewidth=lw, label='ROC curve (area = %0.4f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', linewidth=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('GBDT ROC curve (test)')
plt.legend(loc="lower right")
plt.show()


#########################################################################################################################
# xgboost model
from xgboost.sklearn import XGBClassifier
import seaborn as sns
xgbt = XGBClassifier()
param_grid = {
    'learning_rate':[0.01,0.05,0.1],
    'n_estimators':[100,200,300],
    'max_depth':[6,10,15]
}
xgbt_gr = GridSearchCV(xgbt, param_grid, scoring='roc_auc', cv=4)
xgbt_gr.fit(x_train,y_train)
xgbt_gr.best_params_

# 在训练集上训练XGBoost模型
xgbt = XGBClassifier(n_estimator=200, learning_rate=0.05, max_depth=15)
xgbt.fit(x_train, y_train)
train_predict = xgbt.predict(x_train)
test_predict = xgbt.predict(x_test)
## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('xgboost在训练集上的准确率:',metrics.accuracy_score(y_train,train_predict))
print('xgboost在测试集上的准确率:',metrics.accuracy_score(y_test,test_predict))
print('xgboost训练集的auc：', metrics.roc_auc_score(y_train,train_predict))
print('xgboost测试集的auc：', metrics.roc_auc_score(y_test,test_predict))
## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
print('The confusion matrix result:\n',confusion_matrix_result)
# roc curve
test_est = xgbt.predict(x_train)
fpr_train, tpr_train, thresholds = metrics.roc_curve(test_est, y_train)
test_est = xgbt.predict(x_test)
fpr_test, tpr_test, thresholds = metrics.roc_curve(test_est, y_test)

auc_value = metrics.auc(fpr_train, tpr_train)
plt.figure()
lw = 2
plt.plot(fpr_train, tpr_train, color='darkorange', linewidth=lw, label='ROC curve (area = %0.4f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', linewidth=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('xgboost ROC curve (train)')
plt.legend(loc="lower right")
plt.show()

#利用xgboost进行特征选择
data_features_part = list(x_train.columns)
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False

plt.figure()
plt.title("xgboost feature importance")
sns.barplot(y=data_features_part, x=xgbt.feature_importances_)
plt.subplots_adjust(left=0.4)
plt.show()

#######################模型比较（学习曲线）
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
result_rfc = []
result_gbdt = []
result_xgbt = []
x = df_model.drop(['first_month_discount'], axis=1)
y = df_model['first_month_discount']
# 每隔1000步建立一个随机森林，获得不同sample的得分
for i in range(1000,len(x),1000):
    rfc_learn = ensemble.RandomForestClassifier(criterion='entropy', max_depth=8, max_features=0.3, min_samples_split=12,
                                          n_estimators=15)
    gbm_learn = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70, max_depth=16, min_samples_leaf=10,
                                      min_samples_split=3, max_features='sqrt', random_state=10)
    xgbt_learn = XGBClassifier(n_estimator=200, learning_rate=0.05, max_depth=15)
    score_rfc = cross_val_score(rfc_learn, x.iloc[0:i, :], y[0:i], cv=10).mean()
    score_gbdt = cross_val_score(gbm_learn, x.iloc[0:i, :], y[0:i], cv=10).mean()
    score_xgbt = cross_val_score(xgbt_learn, x.iloc[0:i, :], y[0:i], cv=10).mean()
    result_rfc.append(score_rfc)
    result_gbdt.append(score_gbdt)
    result_xgbt.append(score_xgbt)
# score_max = max(sample_size)
# print('最大得分：{}'.format(score_max),
#       '子树数量为：{}'.format(sample_size.index(score_max)*1000+1))

# 绘制学习曲线
x_axis = np.arange(1000,len(x),1000)
plt.subplot(111)
plt.plot(x_axis, a, 'o-', color='darkturquoise')
plt.plot(x_axis, result_gbdt,'o-', color='lightcoral')
plt.plot(x_axis, result_xgbt,'o-', color='yellow')
plt.legend(['random forest', 'gbdt', 'xgboost'])
plt.show()

a = result_rfc


