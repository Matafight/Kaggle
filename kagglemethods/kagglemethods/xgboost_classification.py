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
    def __init__(self,x,y,metric,metric_proba = False,metric_name='auc',scoring='roc_auc',n_jobs=4,save_model=False,processed_data_version_dir= './',scale_pos_weight=1,if_classification = True):
        super(xgboostClassification_CV,self).__init__(x,y,metric,metric_proba = metric_proba,metric_name= metric_name,scoring=scoring,n_jobs=n_jobs,save_model=save_model,processed_data_version_dir = processed_data_version_dir,if_classification = if_classification)
        #计算负正样本的比例
        self.model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            nthread=n_jobs,
            )

    def cross_validation(self):
        bst_model = super(xgboostClassification_CV,self).cross_validation()
        self.plot_save('xgboostClassification')
        return bst_model


















