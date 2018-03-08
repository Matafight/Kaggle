#_*_coding:utf-8_*_
# 根据函数名 调用对应的函数，工厂模式？

import sys
sys.path.append('..')
from dm_methods.kaggle_methods.lightgbm_regression import lightgbmRegression_CV
from dm_methods.kaggle_methods.ridge import ridge_CV
from sklearn import metrics

import pandas as pd
import numpy as np


df_train = pd.read_csv('./input/processed_train_logged.csv')
df_test = pd.read_csv('./input/processed_test_logged.csv')

df_train = df_train.drop('Id',axis=1)
df_test = df_test.drop('Id',axis=1)
train_label = df_train['label']
df_train = df_train.drop('label',axis=1)
training = df_train.values
testing = df_test.values


method_name = 'lightgbmRegression_CV'
metric = metrics.mean_squared_error
mcls = eval(method_name)(training,train_label,metric = metric,scoring = 'neg_mean_squared_error')
mcls.cross_validation()






