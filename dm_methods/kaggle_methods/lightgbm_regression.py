#_*_coding:utf-8
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from . import log_class
import numpy as np
import matplotlib.pyplot as plt
from . import lightgbm_base

class lightgbmRegression_CV(lightgbm_base.lightgbm_CV):
    def __init__(self,x,y,metric,metric_proba=False,metric_name='l2',scoring = 'neg_mean_squared_error',n_jobs=2):
        super(lightgbmRegression_CV,self).__init__(x,y,metric,metric_proba=metric_proba,metric_name=metric_name,scoring=scoring,n_jobs=n_jobs)
        self.model = LGBMRegressor(silent=True,n_jobs=n_jobs)
   
  


    def cross_validation(self):
        bst_model = super(lightgbmRegression_CV,self).cross_validation()
        self.plot_save()
        return bst_model