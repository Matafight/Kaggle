#_*_coding:utf-8
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from . import log_class
import numpy as np

tunned_num_leaves = [25,31,45]
tunned_max_depth = [10,20,30]

'''
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
class lightgbmClassification_CV:
    def __init__(self,x,y,metric,metric_name='auc',scoring = 'auc'):
        self.x = x
        self.y = y
        #use default values
        self.model = LGBMClassifier(silent=True)
        self.metric = metric
        self.metric_name = metric_name
        self.scoring = scoring 
        self.logger = log_class.log_class('lightgbm')

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
            params['verbosity'] = -1
            lst = lgb.train(params,dtrain)
            pred = lst.predict(valid_x)
            scores.append(self.metric(valid_y,pred))
        msg = 'score {} of cv {}'.format(self.metric.__name__,np.mean(scores))
        print(msg)
        self.logger.add(msg)


    
    def cross_validation(self):
        scoring = self.scoring
        self.modelfit()
        #print('tunning num_leaves...')
        #params = {'num_leaves':tunned_num_leaves}
        #print(self.model.get_params())
        #self.model.set_params(silent = True)
        #gsearch = GridSearchCV(self.model,param_grid=params,scoring=scoring,n_jobs=1,iid=False,cv=3)
        #gsearch.fit(self.x,self.y)
        #self.model.set_params(num_leaves = gsearch.best_params_['num_leaves'])
        #print(gsearch.best_params_['num_leaves'])
        #self.modelfit()
        #self.cv_score()

        self.model.fit(self.x,self.y)
        return self.model