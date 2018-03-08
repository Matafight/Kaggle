#_*_coding:utf-8_*_
# 根据函数名 调用对应的函数，工厂模式？

import sys
sys.path.append('..')
from dm_methods.kaggle_methods.lightgbm_regression import lightgbmRegression_CV
from dm_methods.kaggle_methods.ridge import ridge_CV
from dm_methods.kaggle_methods.xgboost_regression import xgboostRegression_CV
from sklearn import metrics

import pandas as pd
import numpy as np


df_train = pd.read_csv('./input/processed_train_logged.csv')
df_test = pd.read_csv('./input/processed_test_logged.csv')
submission = pd.DataFrame({'id':df_test['Id']})
df_train = df_train.drop('Id',axis=1)
df_test = df_test.drop('Id',axis=1)
train_label = df_train['label']
df_train = df_train.drop('label',axis=1)
training = df_train.values
testing = df_test.values


#method_name = 'lightgbmRegression_CV'
method_name='xgboostRegression_CV'
metric = metrics.mean_squared_error
mcls = eval(method_name)(training,train_label,metric = metric,scoring = 'neg_mean_squared_error')
lgbmodel = mcls.cross_validation()

# generate submission
pred_test = lgbmodel.predict(testing)
pred_test = np.exp(pred_test)
submission['SalePrice'] = pred_test
submission.to_csv(method_name+'.csv',index=False)




