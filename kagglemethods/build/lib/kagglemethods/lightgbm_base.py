#_*_coding:utf-8
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from . import log_class
import numpy as np
import matplotlib.pyplot as plt

tunned_num_leaves = [31,25]
tunned_max_depth = [3,5]
tunned_min_data_in_leaf = [20,15]

class lightgbm_CV(object):
    """
    lightgbm 用于分类或回归任务的交叉验证代码
    """

    def __init__(self,x,y,tunning_params,metric,metric_proba = 0,metric_name='l2',scoring = 'neg_mean_squared_error',n_jobs=2,save_model = 0,processed_data_version_dir='./',labels=None,if_classification=0):
        """
        初始化相关参数

        args:
            x: numpy array
            y: numpy array
            tunning_params: 字典类型，key是待调整参数名,values是候选集合
            metric: sklearn 中的函数，用来在交叉验证中评估验证集上的效果，不过auc 不行，因为auc的参数 不是 (y_true,y_pred) 的形式
                 optional: http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
            metric_proba: 0, 表示 metric 函数是否接受模型输出0-1之间的概率值
            metric_name: 'auc' 这个参数表示 调用lightgbm.cv函数时用来度量的损失函数，可选值为
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
            scoring: 用sklearn自带的 GridSearchCV 时需要的评估函数, 一般是越大越好。默认为 neg_mean_squared_error
                    可选项: 'neg_log_loss' 'roc_auc' ,'neg_mean_squared_error' 等
            n_jobs: 多少个线程,默认为2
            save_model: 1 or 0, 表示是否保存模型,保存路径为 processed_data_version_dir/modules/
            processed_data_version_dir: 存放log 或者保存模型的目录,默认为 ./
            labels: 表示分类的任务一共有多少类，因为有时候样本只有一个类，这时损失函数的计算出出错
            if_classification: 0 or 1，表示是否是分类任务
        """
        import os
        if not os.path.exists(processed_data_version_dir):
            os.mkdir(processed_data_version_dir)
        self.x = x
        self.y = y
        self.tunning_params = tunning_params
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
        self.labels = labels
        self.if_classification = if_classification

    def modelfit(self):
        #return the best n_estimators
        params = self.model.get_params()
        params['verbosity'] = -1
        num_rounds = 500
        early_stopping = 10
        dtrain = lgb.Dataset(self.x,label=self.y)
        ## 第一个参数其实是 booster参数，其实 silent 和 n_jobs都不是booster参数
        lgbm = lgb.cv(params,dtrain,num_boost_round=num_rounds,nfold=5,metrics = self.metric_name,early_stopping_rounds=early_stopping,verbose_eval=False)
        best_rounds = len(lgbm[self.metric_name+'-mean'])
        self.model.set_params(n_estimators=best_rounds)
        self.model.fit(self.x,self.y)
        if self.metric_proba == 0:
            pred = self.model.predict(self.x)
        else:
            pred = self.model.predict_proba(self.x)
        self.train_scores.append(self.metric(self.y,pred))

    def cv_score(self):
        # k-fold cross score
        if self.if_classification == 1:
            kf = StratifiedKFold(n_splits = 5,shuffle=True,random_state=2018)
        else:
            kf = KFold(n_splits=5,shuffle=True,random_state=2018)
        scores = []
        params = self.model.get_params()
        for train_ind,valid_ind in kf.split(self.x,self.y):
            train_x,valid_x = self.x[train_ind],self.x[valid_ind]
            train_y,valid_y = self.y[train_ind],self.y[valid_ind]
            dtrain = lgb.Dataset(train_x,label = train_y)
            params['verbosity'] = -1
            lst = lgb.train(params,dtrain)
            if self.metric_proba == 0:
                pred = lst.predict(valid_x)
            else:
                pred = lst.predict_proba(valid_x)

            #if self.labels == None:
            scores.append(self.metric(valid_y,pred))
            #else:
            #    scores.append(self.metric(valid_y,pred,labels=self.labels))
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
        if self.save_model==1:
            #save model here
            from sklearn.externals import joblib
            if not os.path.exists(npath+'/modules'):
                os.mkdir(npath+'/modules')
            joblib.dump(self.model,npath+'/modules/'+name+"_"+cur_time+".pkl")




    def cross_validation(self):
        scoring = self.scoring
        for param_item in self.tunning_params.keys():
            print('tunning {} ...'.format(param_item))
            params = {param_item:self.tunning_params[param_item]}
            print(self.model.get_params())
            gsearch = GridSearchCV(self.model,param_grid=params,scoring=scoring,n_jobs=1,iid=False,cv=3)
            gsearch.fit(self.x,self.y)
            #这是一种将变量作为参数名的方法
            to_set = {param_item:gsearch.best_params_[param_item]}
            self.model.set_params(**to_set)
            print(gsearch.best_params_)
            self.modelfit()
            print('best_num_round after tunning para: {}'.format(self.model.get_params()['n_estimators']))
            self.cv_score()

        ## 保存最终的参数
        params = self.model.get_params()
        self.logger.add(params,ifdict=1)
        #用新参数重新训练一遍
        self.model.fit(self.x,self.y)
        
        return self.model
