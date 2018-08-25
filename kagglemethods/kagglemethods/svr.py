#_*_coding:utf-8_*_
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from . import learning_methods
import numpy as np

from sklearn.svm import SVR

tunned_C =[0.5,1,1.3]

class SVR_CV(learning_methods.learning_methods):
    def __init__(self,x,y,metric,scoring='neg_log_loss',n_jobs=2,save_model=False,processed_data_version_dir='./'):
        super(SVR_CV,self).__init__(x,y,metric,scoring=scoring,save_model=save_model,processed_data_version_dir=processed_data_version_dir)
        #default kernel is rbf
        self.model = SVR(C = 1.0)
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
        self.plot_save('SVR')
        return self.model