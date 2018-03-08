#_*_coding:utf-8_*_
# 根据函数名 调用对应的函数，工厂模式？

import sys
sys.path.append('..')
from dm_methods.kaggle_methods.lightgbm_regression import lightgbmRegression_CV
from dm_methods.kaggle_methods.ridge import ridge_cv
import pandas as pd
import numpy as np


df_train = pd.read_csv('./input/processed_train_logged.csv')
df_test = pd.read_csv('./input/processed_test_logged.csv')

df_train = df_train.drop('Id',axis=1)
df_test = df_test.drop('Id',axis=1)





