#_*_coding:utf-8_*_
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
from . import learning_methods_classification

tunned_max_depth = [5,10,15]
tunned_min_sample_split = [1,2,4,8]
#add more feautre tunning in the future

#应该写一个基类，然后继承该基类
class RandomForest_CV(learning_methods_classification.learning_methods):
    def __init__(self,x,y,metric):
        super(RandomForest_CV,self).__init__(x,y,metric)
        self.model = RandomForestClassifier()


    #scoring:neg_log_loss
    def cross_validation(self,scoring='neg_log_loss'):
        params = {'max_depth':tunned_max_depth}
        gsearch = GridSearchCV(estimator=self.model,param_grid=params,scoring=scoring,n_jobs=1,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(max_depth= gsearch.best_params_['max_depth'])
        print('best max_depth for rf:{}'.format(gsearch.best_params_['max_depth']))
        self.cv_score()

        params = {'min_sample_split':tunned_min_sample_split}
        gsearch = GridSearchCV(estimator = self.model,param_grid = params,scoring=scoring,n_jobs=1,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(min_sample_split = gsearch.best_params_['min_sample_split'])
        print('best min_sample_split for rf:{}'.format(gsearch.best_params_['min_sample_split']))
        self.cv_score()

        
        self.model.fit(self.x,self.y)
        return self.model

