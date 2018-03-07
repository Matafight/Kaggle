#_*_coding:utf-8_*_
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
from . import learning_methods_classification
from . import log_class
tunned_n_estimators = [120,200,300]
tunned_max_depth = [5,10,15,25]
tunned_min_samples_split = [2,4,10,15]
tunned_min_samples_leaf =[1,2,5,10]
tunned_max_features = ['sqrt','log2','auto',None]
#add more feautre tunning in the future

#应该写一个基类，然后继承该基类
class RandomForest_CV(learning_methods_classification.learning_methods):
    def __init__(self,x,y,metric):
        super(RandomForest_CV,self).__init__(x,y,metric)
        self.model = RandomForestClassifier()
        self.logger = log_class.log_class('random_forest')

    
    def cv_score(self):
        ret = super(RandomForest_CV,self).cv_score()
        print(ret)
        #add logger here
        self.logger.add(ret)
    def train_score(self):
        ret = super(RandomForest_CV,self).train_score()
        print(ret)
        self.logger.add(ret)

    #scoring:neg_log_loss
    def cross_validation(self,scoring='neg_log_loss'):
        params = {'n_estimators':tunned_n_estimators}
        gsearch = GridSearchCV(estimator=self.model,param_grid=params,scoring=scoring,n_jobs=1,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(n_estimators= gsearch.best_params_['n_estimators'])
        print('best n_estimators for rf:{}'.format(gsearch.best_params_['n_estimators']))
        self.cv_score()
  



        params = {'max_depth':tunned_max_depth}
        gsearch = GridSearchCV(estimator=self.model,param_grid=params,scoring=scoring,n_jobs=1,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(max_depth= gsearch.best_params_['max_depth'])
        print('best max_depth for rf:{}'.format(gsearch.best_params_['max_depth']))
        self.cv_score()
  

        params = {'min_samples_split':tunned_min_samples_split}
        gsearch = GridSearchCV(estimator = self.model,param_grid = params,scoring=scoring,n_jobs=1,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(min_samples_split = gsearch.best_params_['min_samples_split'])
        print('best min_samples_split for rf:{}'.format(gsearch.best_params_['min_samples_split']))
        self.cv_score()

        params = {'min_samples_leaf':tunned_min_samples_leaf}
        gsearch = GridSearchCV(estimator = self.model,param_grid = params,scoring=scoring,n_jobs=1,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(min_samples_leaf = gsearch.best_params_['min_samples_leaf'])
        print('best min_samples_leaf for rf:{}'.format(gsearch.best_params_['min_samples_leaf']))
        self.cv_score()

        params = {'max_features':tunned_max_features}
        gsearch = GridSearchCV(estimator = self.model,param_grid = params,scoring=scoring,n_jobs=1,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(max_features = gsearch.best_params_['max_features'])
        print('best max_features for rf:{}'.format(gsearch.best_params_['max_features']))
        self.cv_score()

        
        self.model.fit(self.x,self.y)
        self.train_score()
        return self.model

