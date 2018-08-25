#_*_coding:utf-8
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from . import log_class
import numpy as np
from . import lightgbm_base


class lightgbmClassification_CV(lightgbm_base.lightgbm_CV):
    def __init__(self,x,y,metric,metric_proba = False,metric_name='auc',scoring = 'auc',n_jobs=2,save_model=False,processed_data_version_dir='./'):
        super(lightgbmClassification_CV,self).__init__(x,y,metric,metric_proba,metric_name=metric_name,scoring=scoring,n_jobs=n_jobs,save_model=save_model,processed_data_version_dir = processed_data_version_dir)
        self.model = LGBMClassifier(silent=True,n_jobs=n_jobs)

        

      
    def cross_validation(self):
        bst_model = super(lightgbmClassification_CV,self).cross_validation()
        self.plot_save('lightgbmClassification')
        return bst_model