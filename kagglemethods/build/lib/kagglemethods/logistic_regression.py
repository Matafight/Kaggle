#_*_coding:utf-8_*_
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import numpy as np
from . import learning_methods
from . import log_class
tunned_C = [1]

class LogisticRegression_CV(learning_methods.learning_methods):
    def __init__(self,x,y,metric,metric_proba = False,labels = [0,1],scoring='neg_mean_squared_error',n_jobs=2,save_model=False,processed_data_version_dir='./'):
        super(LogisticRegression_CV,self).__init__(x,y,metric,metric_proba=metric_proba,labels = labels,scoring = scoring,save_model=save_model,processed_data_version_dir=processed_data_version_dir)
        self.model = LogisticRegression(C = 1.0,class_weight='balanced')
        self.n_jobs = n_jobs



    def cross_validation(self):
        scoring = self.scoring
        #self.cv_score()
        #self.train_score()

        #params = {'C':tunned_C}
        #gsearch = GridSearchCV(estimator=self.model,param_grid=params,scoring=scoring,n_jobs=1,iid=False,cv=3)
        #gsearch.fit(self.x,self.y)
        #self.model.set_params(C= gsearch.best_params_['C'])
        #print('best C for lr:{}'.format(gsearch.best_params_['C']))
        #self.cv_score()
        #self.model.fit(self.x,self.y)
        #return self.model

        #use the model_selection cross_val_score function
        scores = []
        for para_C in tunned_C:
            self.model.set_params(C = para_C)
            scores.append(np.mean(cross_val_score(self.model,self.x,self.y,scoring = scoring,cv=3)))
        #get max score
        best_C = scores.index(max(scores))
        print('best C for lr:{}'.format(tunned_C[best_C]))
        self.model.set_params(C = tunned_C[best_C])
        self.model.fit(self.x,self.y)
        #self.cv_score()
        #self.train_score()
        #self.plot_save("logisticRegression")
        return self.model

