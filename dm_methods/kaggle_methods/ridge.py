from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
from . import learning_methods_regression
tunned_alpha = [0.1,0.3,0.5,1,1.2,1.5,2]

class ridge_CV(learning_methods_regression.learning_methods):
    #metric is a function parameter
    def __init__(self,x,y,metric,scoring = 'neg_mean_squared_error'):
        super(ridge_cv,self).__init__(x,y,metric)
        #normalize 
        self.model = Ridge(alpha=1,normalize=True)
        self.scoring = scoring 


    #scoring : neg_mean_squared_error 
    def cross_validation(self):
        scoring = self.scoring
        params = {'alpha':tunned_alpha}
        gsearch = GridSearchCV(estimator=self.model,param_grid=params,scoring=scoring,n_jobs=1,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(alpha = gsearch.best_params_['alpha'])
        print('best alpha for ridge:{}'.format(gsearch.best_params_['alpha']))
        self.cv_score()
        self.model.fit(self.x,self.y)
        return self.model

if __name__ == '__main__':
    x= None
    y = None
    metrics = None
    obj = ridge_cv(x,y,metrics)
        
       
    