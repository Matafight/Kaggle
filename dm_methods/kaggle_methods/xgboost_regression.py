#_*_ coding:utf-8
'''
Parameters needed:
1. metrics
2. x
3. y
'''
import xgboost as xgb
import numpy as np
from xgboost.sklearn import  XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import metrics
import time

#parameters to be tunned
tunned_max_depth = [3,5,7,9]
tunned_learning_rate =[0.01,0.015,0.025,0.05,0.1]  # aka eta in xgboost
tunned_min_child_weight = [1,3,5,7]
tunned_gamma = [0.05,0.1,0.3,0.5,0.7,0.9,1]
tunned_colsample_bytree = [0.6,0.7,0.8,1]

# 还需要添加subsample,lambda,alpha等参数


#define a decorator function to record the running time

class xgboost_regression_cv:
    def __init__(self,x,y,metrics):
        self.x =  x
        self.y = y
        # define the algorithm using default parameters
        self.model = XGBRegressor(
            learning_rate= 0.5,
            max_depth = 20,
            n_estimators = 100,
            min_child_weight = 1,
            gamma = 0,
            objective='reg:linear',
            nthread=4,
            )
        self.cv_folds = 5
        self.early_stopping_rounds = 50
        self.metrics = metrics
    
    # get the best numrounds after changing a parameter
    def modelfit(self):
        xgb_param = self.model.get_xgb_params()
        dtrain = xgb.DMatrix(self.x,label = self.y)
        cvresult = xgb.cv(xgb_param,dtrain,num_boost_round=500,nfold=self.cv_folds,metrics='rmse',early_stopping_rounds=self.early_stopping_rounds)
        self.model.set_params(n_estimators=cvresult.shape[0])
        self.model.fit(dtrain,self.y,eval_metric='rmse')

    def cv_score(self):
        # 5-fold crossvalidation error
        kf = KFold(n_splits = 5)
        score = []
        params = self.model.get_xgb_params()
        for train_ind,test_ind in kf.split(self.x):
            train_valid_x,train_valid_y = self.x[train_ind],self.y[train_ind]
            test_valid_x,test_valid_y = self.x[test_ind],self.y[test_ind]
            dtrain = xgb.DMatrix(train_valid_x,label = train_valid_y)
            dtest = xgb.DMatrix(test_valid_x)
            pred_model = xgb.train(params,dtrain,num_boost_round=int(params['n_estimators']))
            pred_test = pred_model.predict(dtest)
            score.append(metrics.mean_squared_error(test_valid_y,pred_test))
        print('final mse on cv:')
        print(np.mean(score))

    def cross_validation(self,scoring='neg_mean_squared_error'):
        self.modelfit()        
        print('tunning learning_rate...')
        params = {'learning_rate':tunned_learning_rate}
        gsearch = GridSearchCV(estimator=self.model,param_grid=params,scoring=scoring,n_jobs=1,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(learning_rate = gsearch.best_params_['learning_rate'])
        print(gsearch.best_params_)
        
        self.modelfit()
        print('best_num_round after tunning para: {}'.format(self.model.get_params()['n_estimators']))
        self.cv_score()

        print('tunning max_depth...')
        depth_params = {'max_depth':tunned_max_depth}
        gsearch = GridSearchCV(estimator= self.model,param_grid =depth_params,scoring=scoring,n_jobs=1,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(max_depth = gsearch.best_params_['max_depth'])
        print(gsearch.best_params_)

        self.modelfit()
        print('best_num_round after tunning para: {}'.format(self.model.get_params()['n_estimators']))
        self.cv_score()


        print('tunning min_child_weight...')
        min_child_weight_params = {'min_child_weight':tunned_min_child_weight}
        gsearch = GridSearchCV(estimator=self.model,param_grid = min_child_weight_params,scoring=scoring,n_jobs=1,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(min_child_weight = gsearch.best_params_['min_child_weight'])
        print(gsearch.best_params_)

        self.modelfit()
        print('best_num_round after tunning para: {}'.format(self.model.get_params()['n_estimators']))
        self.cv_score()

        self.model.fit(self.x,self.y)
        return self.model

        








        

    
    




