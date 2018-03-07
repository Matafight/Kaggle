#_*_coding:utf-8
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from . import log_class
import numpy as np
tunned_num_leaves = [25,31,45]
tunned_max_depth = [10,20,30]


class lightgbm_CV:
    def __init__(self,x,y,metric):
        self.x = x
        self.y = y
        #use default values
        self.model = LGBMClassifier()
        self.metric = metric
        self.logger = log_class.log_class('lightgbm')

    def modelfit(self):
        #return the best n_estimators
        params = self.model.get_params()
        num_rounds = 500
        early_stopping = 10
        dtrain = lgb.Dataset(self.x,label=self.y)
        lgbm = lgb.cv(params,dtrain,num_boost_round=num_rounds,nfold=5,metrics = 'auc',early_stopping_rounds=early_stopping)
        print(lgbm)
        best_rounds = len(lgbm['auc-mean'])
        self.model.set_params(n_estimators=best_rounds)
        #记录一下训练集score
        #self.model.fit(self.x,self.y)
        #train_score = self.metric(self.y,self.model.predict_proba())

    def cv_score(self):
        # k-fold cross score
        scores = []
        params = self.model.get_params()
        for train_ind,valid_ind in KFold(n_splits=5).split(self.x):
            train_x,valid_x = self.x[train_ind],self.x[valid_ind]
            train_y,valid_y = self.y[train_ind],self.y[valid_ind]
            dtrain = lgb.Dataset(train_x,label = train_y)
            #dvalid = lgb.Dataset(valid_x,label=valid_y)
            lst = lgb.train(params,dtrain)
            pred = lst.predict(valid_x)
            scores.append(self.metric(valid_y,pred))
        msg = 'score {} of cv {}'.format(self.metric.__name__,np.mean(scores))
        print(msg)
        self.logger.add(msg)


    
    def cross_validation(self,scoring = 'neg_log_loss'):
        self.modelfit()
        print('tunning num_leaves...')
        params = {'num_leaves':tunned_num_leaves}
        gsearch = GridSearchCV(self.model,param_grid=params,scoring=scoring,n_jobs=1,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(num_leaves = gsearch.best_params_['num_leaves'])
        print(gsearch.best_params_['num_leaves'])
        self.modelfit()
        self.cv_score()

        self.model.fit(self.x,self.y)
        return self.model