#_*_coding:utf-8_*_
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from . import learning_methods
import numpy as np

from sklearn.svm import SVC

tunned_C =[0.5,1,1.3]

class SVC_CV(learning_methods.learning_methods):
    def __init__(self,x,y,metric,metric_proba = True,labels = [0,1],scoring='neg_log_loss',n_jobs=2,save_model=False):
        super(SVC_CV,self).__init__(x,y,metric,metric_proba = metric_proba,labels = labels,scoring=scoring,save_model=save_model)
        #default kernel is rbf
        self.model = SVC(C = 1.0,probability=True)
        self.n_jobs = n_jobs


    def cross_validation(self):
        scoring = self.scoring
        self.train_score()
        self.cv_score()
        params = {'C':tunned_C}
        gsearch = GridSearchCV(estimator=self.model,param_grid=params,scoring=scoring,n_jobs=self.n_jobs,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(C= gsearch.best_params_['C'])
        print('best C for svc:{}'.format(gsearch.best_params_['C']))
        self.cv_score()
        self.train_score()
        self.plot_save("SVC")
        return self.model