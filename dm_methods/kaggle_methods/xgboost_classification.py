#_*_ coding:utf-8

import xgboost as xgb
import numpy as np
from xgboost.sklearn import  XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import metrics
import time
from . import xgboost_base


class xgboostClassification_CV(xgboost_base.xgboost_CV):
    def __init__(self,x,y,metric,metric_proba = True,metric_name='auc',scoring='roc_auc',n_jobs=2,save_model=False,processed_data_version_dir= './'):
        super(xgboostClassification_CV,self).__init__(x,y,metric,metric_proba = metric_proba,metric_name= metric_name,scoring=scoring,n_jobs=n_jobs,save_model=save_model,processed_data_version_dir = processed_data_version_dir)
        self.model = XGBClassifier(
            learning_rate= 0.5,
            max_depth = 20,
            n_estimators = 100,
            min_child_weight = 1,
            gamma = 0,
            objective='binary:logistic',
            nthread=n_jobs,
            )

    def cross_validation(self):
        bst_model = super(xgboostClassification_CV,self).cross_validation()
        self.plot_save('xgboostClassification')
        return bst_model

        








        

    
    




