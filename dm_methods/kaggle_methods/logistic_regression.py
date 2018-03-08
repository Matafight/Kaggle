#_*_coding:utf-8_*_
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import numpy as np
from . import learning_methods_classification
from . import log_class
tunned_C = [0.2,0.5,1,1.5]

#应该写一个基类，然后继承该基类
class LogisticRegression_CV(learning_methods_classification.learning_methods):
    def __init__(self,x,y,metric,scoring='auc'):
        super(LogisticRegression_CV,self).__init__(x,y,metric)
        self.model = LogisticRegression(C = 1.0,class_weight='balanced')
        self.logger = log_class.log_class("logistic-regression")
        self.scoring = scoring
    def cv_score(self):
        ret = super(LogisticRegression_CV,self).cv_score()
        print(ret)
        #add logger here
        self.logger.add(ret)
    def train_score(self):
        ret = super(LogisticRegression_CV,self).train_score()
        print(ret)
        self.logger.add(ret)
    #scoring:neg_log_loss
    def cross_validation(self):
        scoring = self.scoring
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
        self.cv_score()
        self.model.set_params(C = tunned_C[best_C])
        self.model.fit(self.x,self.y)
        self.train_score()
        return self.model

