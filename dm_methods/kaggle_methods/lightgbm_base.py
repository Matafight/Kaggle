#_*_coding:utf-8
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from . import log_class
import numpy as np
import matplotlib.pyplot as plt

tunned_num_leaves = [25,31,45,70,100]
tunned_max_depth = [-1,20,30,60]
tunned_min_data_in_leaf = [20,40,60,100]

'''
optional: http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
metric: is a function , it's used for evaluate the crossvalidation score

sklearn.metrics.auc is not supported, since the auc function's parameter is not in the form of (y_true,y_pred)

scoring para in cross_validation: it's the criterion to choose the best para in cross validation
it's sklearn parameters, can be 'neg_log_loss' 'roc_auc' ,'neg_mean_squared_error' etc...

metric_name: str, special parameter for xgboost and lightgbm when they doing cross validation to find the best num_rounds
options :
if None, metric corresponding to specified application will be used
1. l1, absolute loss, alias=mean_absolute_error, mae, regression_l1
2. l2, square loss, alias=mean_squared_error, mse, regression_l2, regression
3. l2_root, root square loss, alias=root_mean_squared_error, rmse
4. quantile, Quantile regression
5. mape, MAPE loss, alias=mean_absolute_percentage_error
6. huber, Huber loss
7. fair, Fair loss
8. poisson, negative log-likelihood for Poisson regression
9. gamma, negative log-likelihood for Gamma regression
10. gamma_deviance, residual deviance for Gamma regression
11. tweedie, negative log-likelihood for Tweedie regression
12. ndcg, NDCG
13. map, MAP, alias=mean_average_precision
14. auc, AUC
15. binary_logloss, log loss, alias=binary
16. binary_error, for one sample: 0 for correct classification, 1 for error classification
17. multi_logloss, log loss for mulit-class classification, alias=multiclass, softmax, multiclassova, multiclass_ova, ova, ovr
18. multi_error, error rate for mulit-class classification
19. xentropy, cross-entropy (with optional linear weights), alias=cross_entropy
20. xentlambda, "intensity-weighted" cross-entropy, alias=cross_entropy_lambda
21. kldiv, Kullback-Leibler divergence, alias=kullback_leibler
support multi metrics, separated by ,

'''
class lightgbm_CV(object):
    def __init__(self,x,y,metric,metric_proba = False,metric_name='l2',scoring = 'neg_mean_squared_error',n_jobs=2,save_model = False,processed_data_version_dir='./'):
        import os
        if not os.path.exists(processed_data_version_dir):
            os.mkdir(processed_data_version_dir)
        self.x = x
        self.y = y
        #use default values
        self.model = None
        self.metric = metric
        self.metric_proba = metric_proba        
        self.metric_name = metric_name
        self.scoring = scoring 
        self.logger = log_class.log_class('lightgbm',top_level=processed_data_version_dir)
        self.n_jobs=n_jobs
        self.save_model = save_model
        self.train_scores=[]
        self.cv_scores = []

        
        self.path = processed_data_version_dir

    def modelfit(self):
        #return the best n_estimators
        params = self.model.get_params()
        params['verbosity'] = -1
        num_rounds = 500
        early_stopping = 10
        dtrain = lgb.Dataset(self.x,label=self.y)
        lgbm = lgb.cv(params,dtrain,num_boost_round=num_rounds,nfold=5,metrics = self.metric_name,early_stopping_rounds=early_stopping)
        best_rounds = len(lgbm[self.metric_name+'-mean'])
        self.model.set_params(n_estimators=best_rounds)
        #查看设定了新的参数之后lgb的训练和验证误差分别是多少，可以把曲线图画出来。最好先保存起来，然后等所有的结果出来后画在一张图上
        #要的是训练误差和验证误差随着迭代次数的变化曲线图
        #有一个问题就是 验证集从哪里来
        
        #记录一下训练集score
        self.model.fit(self.x,self.y)
        if self.metric_proba == False:
            pred = self.model.predict(self.x)
        else:
            pred = self.model.predict_proba(self.x) 
        self.train_scores.append(self.metric(self.y,pred))

    def cv_score(self):
        # k-fold cross score
        scores = []
        params = self.model.get_params()
        for train_ind,valid_ind in KFold(n_splits=5).split(self.x):
            train_x,valid_x = self.x[train_ind],self.x[valid_ind]
            train_y,valid_y = self.y[train_ind],self.y[valid_ind]
            dtrain = lgb.Dataset(train_x,label = train_y)
            params['verbosity'] = -1
            lst = lgb.train(params,dtrain)
            if self.metric_proba == False:
                pred = lst.predict(valid_x)
            else:
                pred = lst.predict_proba(valid_x)
            scores.append(self.metric(valid_y,pred))
        mean_cv_score = np.mean(scores)
        self.cv_scores.append(mean_cv_score)
        msg = 'score {} of cv {}'.format(self.metric.__name__,mean_cv_score)
        print(msg)
        #self.logger.add(msg)


    def plot_save(self,name='lightgbmRegression'):
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
        npath = self.path
        if not os.path.exists(npath+'/curve'):
            os.mkdir(npath+'/curve')
        import time
        cur_time = time.strftime("%Y-%m-%d-%H-%M",time.localtime())
        fig.savefig(npath+'/curve/'+name+'_'+cur_time+'_train_cv.png')

        #save train score and cv score
        str_train_score ="train score sequence "+" ".join([str(item) for item in self.train_scores])
        str_cv_score = "cv score sequence"+ " ".join([str(item) for item in self.cv_scores])
        self.logger.add(str_train_score)
        self.logger.add(str_cv_score)

        #determine if save model
        if self.save_model:
            #save model here
            from sklearn.externals import joblib
            if not os.path.exists(npath+'/modules'):
                os.mkdir(npath+'/modules')
            joblib.dump(self.model,npath+'/modules/'+name+"_"+cur_time+".pkl")

            



    def cross_validation(self):
        scoring = self.scoring

        self.modelfit()
        self.cv_score()

        print('tunning num_leaves...')
        params = {'num_leaves':tunned_num_leaves}
        print(self.model.get_params())
        self.model.set_params(silent = True)
        gsearch = GridSearchCV(self.model,param_grid=params,scoring=scoring,n_jobs=self.n_jobs,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(num_leaves = gsearch.best_params_['num_leaves'])
        print(gsearch.best_params_['num_leaves'])

        self.modelfit()
        self.cv_score()

        print('tunning max_depth...')
        params = {'max_depth':tunned_max_depth}
        gsearch = GridSearchCV(self.model,param_grid=params,scoring=scoring,n_jobs=self.n_jobs,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(max_depth = gsearch.best_params_['max_depth'])
        print(gsearch.best_params_['max_depth'])

        self.modelfit()
        self.cv_score()

        print('tunning min_data_in_leaf...')  
        params = {'min_data_in_leaf':tunned_min_data_in_leaf}
        gsearch = GridSearchCV(self.model,param_grid=params,scoring=scoring,n_jobs=self.n_jobs,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(min_data_in_leaf = gsearch.best_params_['min_data_in_leaf'])
        print(gsearch.best_params_['min_data_in_leaf'])

        self.modelfit()
        self.cv_score()

        self.model.fit(self.x,self.y)
        return self.model