#_*_ coding:utf-8

import xgboost as xgb
import numpy as np
from xgboost.sklearn import  XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import metrics
import time
import matplotlib.pyplot as plt
#parameters to be tunned
#tunned_max_depth = [3,5,7,9]
#tunned_learning_rate =[0.01,0.015,0.025,0.05,0.1]  # aka eta in xgboost
#tunned_min_child_weight = [1,3,5,7]
#tunned_gamma = [0.05,0.1,0.3,0.5,0.7,0.9,1]
#tunned_colsample_bytree = [0.6,0.7,0.8,1]

tunned_max_depth = [3]
tunned_learning_rate =[0.01,0.015]  # aka eta in xgboost
tunned_min_child_weight = [1,3]
tunned_gamma = [0.05]
tunned_colsample_bytree = [0.6]


# 还需要添加subsample,lambda,alpha等参数


#define a decorator function to record the running time
'''

metric_name parameter options:
1. 'rmse'
2. 'mae'
3. 'logloss',  负的对数似然
4. 'error',二类分类错误率，输出大于0.5时为正例，否则负例
5. 'error@t',同上，不过指定正负例阈值
6. 'merror',多类分类错误率
7. 'mlogloss',多类的logloss
8. 'auc'
9. 'ndcg',[Normalized Discounted Cumulative Gain][3]
10. 'map',[Mean average precision][4]
it is used in xgb.cv function
'''
class xgboost_CV(object):
    def __init__(self,x,y,metric,metric_proba = False,metric_name='auc',scoring='roc_auc'):
        '''
        metric_proba indicates if the metric need the probability to calculate the score
        '''
        #default metrics should be metrics.log_loss
        self.x =  x
        self.y = y
        # define the algorithm using default parameters
        self.model = None
        self.cv_folds = 5
        self.early_stopping_rounds = 50
        self.metric = metric
        self.metric_proba = metric_proba
        self.metric_name = metric_name
        self.scoring = scoring 

        self.train_scores = []
        self.cv_scores = []

    def plot_save(self,name='learning_method'):
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(self.train_scores)
        ax1.set_xlabel('train_scores')
        ax1.set_ylabel(self.metric.__name__)
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(self.cv_scores)
        ax2.set_xlabel('cv scores')

        #save 
        import os
        if not os.path.exists('./curve'):
            os.mkdir('./curve')
        import time
        cur_time = time.strftime("%Y-%m-%d-%H-%M",time.localtime())
        fig.savefig('./curve/'+name+'_'+cur_time+'_train_cv.png')        
    
    # get the best numrounds after changing a parameter
    def modelfit(self):
        xgb_param = self.model.get_xgb_params()
        dtrain = xgb.DMatrix(self.x,label = self.y)
        cvresult = xgb.cv(xgb_param,dtrain,num_boost_round=500,nfold=self.cv_folds,metrics=self.metric_name,early_stopping_rounds=self.early_stopping_rounds)
        self.model.set_params(n_estimators=cvresult.shape[0])
        self.model.fit(self.x,self.y)

        #calculate train score
        if self.metric_proba == False:
            pred = self.model.predict(self.x)
        else:
            pred = self.model.predict_proba(self.x)
        self.train_scores.append(self.metric(self.y,pred))


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
            if self.metric_proba == False:
                pred_test = pred_model.predict(dtest)
            else:
                pred_test = pred_model.predict_proba(dtest)
            score.append(self.metric(test_valid_y,pred_test))
        mean_cv_score = np.mean(score)
        self.cv_scores.append(mean_cv_score)
        print('final {} on cv:{}'.format(self.metric.__name__,mean_cv_score))


    def cross_validation(self):
        scoring = self.scoring 
        self.modelfit()
        self.cv_score()

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
        self.model.set_params(max_depth=gsearch.best_params_['max_depth'])
        print(gsearch.best_params_)

        self.modelfit()
        print('best_num_round after tunning para: {}'.format(self.model.get_params()['n_estimators']))
        self.cv_score()


        print('tunning min_child_weight...')
        min_child_weight_params = {'min_child_weight':tunned_min_child_weight}
        gsearch = GridSearchCV(estimator=self.model,param_grid = min_child_weight_params,scoring=scoring,n_jobs=1,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(max_depth=gsearch.best_params_['min_child_weight'])
        print(gsearch.best_params_)

        self.modelfit()
        print('best_num_round after tunning para: {}'.format(self.model.get_params()['n_estimators']))
        self.cv_score()

        self.model.fit(self.x,self.y)
        return self.model

        








        

    
    




