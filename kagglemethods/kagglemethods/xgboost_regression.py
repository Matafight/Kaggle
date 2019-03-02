#_*_ coding:utf-8
'''
Parameters needed:
1. metrics
2. x
3. y
'''
import xgboost as xgb
import numpy as np
from xgboost.sklearn import  XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import metrics
#import xgboost_base
from . import xgboost_base
import time


class xgboostRegression_CV(xgboost_base.xgboost_CV):
    def __init__(self,x,y,metric,metric_proba = False,metric_name='rmse',scoring='neg_mean_squared_error',n_jobs=2,save_model=False,processed_data_version_dir = './',if_classification=False):
        super(xgboostRegression_CV,self).__init__(x,y,metric,metric_proba = metric_proba,metric_name=metric_name,scoring = scoring,n_jobs=n_jobs,save_model=save_model,processed_data_version_dir=processed_data_version_dir,if_classification = if_classification)
        self.model = XGBRegressor(
            #learning_rate= 0.5,
            max_depth = 6,
            #n_estimators = 100,
            min_child_weight = 1,
            gamma = 0,
            objective='reg:linear',
            nthread=n_jobs,
            )

    def cross_validation(self):
        bst_model = super(xgboostRegression_CV,self).cross_validation()
        self.plot_save('xgboostRegression')
        return bst_model

















