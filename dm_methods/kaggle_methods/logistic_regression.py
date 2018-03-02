#_*_coding:utf-8_*_
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
from . import learning_methods

tunned_C = [0.2,0.5,1,1.5]

#应该写一个基类，然后继承该基类
class LogisticRegression_CV(learning_methods.learning_methods):
    def __init__(self,x,y,metric):
        super(LogisticRegression_CV,self).__init__(x,y,metric)
        self.model = LogisticRegression(C = 1.0)


    #scoring:neg_log_loss
    def cross_validation(self,scoring='neg_log_loss'):
        params = {'C':tunned_C}
        gsearch = GridSearchCV(estimator=self.model,param_grid=params,scoring=scoring,n_jobs=1,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(C= gsearch.best_params_['C'])
        print('best C for lr:{}'.format(gsearch.best_params_['C']))
        self.cv_score()
        self.model.fit(self.x,self.y)
        return self.model