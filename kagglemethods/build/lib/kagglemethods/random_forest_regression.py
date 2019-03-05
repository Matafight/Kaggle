#_*_coding:utf-8_*_
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
from . import learning_methods
from . import log_class
tunned_n_estimators = [120,300,500,800,1200]
tunned_max_depth = [None,5,10,15,25,30]
tunned_min_samples_split = [2,4,10,15,100]
tunned_min_samples_leaf =[1,2,5,10]
tunned_max_features = ['sqrt','log2','auto',None]
#add more feautre tunning in the future

class RandomForestRegression_CV(learning_methods.learning_methods):
    """
    随机森林交叉验证
    """
    def __init__(self,x,y,metric,scoring = 'neg_mean_squared_error',n_jobs=3,save_model = False,processed_data_version_dir='./'):
        """
        初始化相关参数

        args:
            x: numpy array
            y: numpy array
            metric: sklearn 中的函数，用来在交叉验证中评估验证集上的效果，不过auc 不行，因为auc的参数 不是 (y_true,y_pred) 的形式
                 optional: http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
            metric_proba: False, 表示 metric 函数是否接受模型输出0-1之间的概率值
            scoring: 用sklearn自带的 GridSearchCV 时需要的评估函数, 一般是越大越好。默认为 neg_mean_squared_error
                    可选项: 'neg_log_loss' 'roc_auc' ,'neg_mean_squared_error' 等
            n_jobs: 多少个线程,默认为3
            save_model: True or False, 表示是否保存模型,保存路径为 processed_data_version_dir/modules/
            processed_data_version_dir: 存放log 或者保存模型的目录,默认为 ./ 
        """
        super(RandomForestRegression_CV,self).__init__(x,y,metric,scoring=scoring,save_model=save_model,processed_data_version_dir=processed_data_version_dir)
        self.model = RandomForestRegressor(n_jobs=n_jobs)
        self.n_jobs = n_jobs

    


    def cross_validation(self):
        scoring = self.scoring 
        self.train_score()
        self.cv_score()

        params = {'n_estimators':tunned_n_estimators}
        gsearch = GridSearchCV(estimator=self.model,param_grid=params,scoring=scoring,n_jobs=self.n_jobs,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(n_estimators= gsearch.best_params_['n_estimators'])
        print('best n_estimators for rf:{}'.format(gsearch.best_params_['n_estimators']))

        self.cv_score()
        self.train_score()



        params = {'max_depth':tunned_max_depth}
        gsearch = GridSearchCV(estimator=self.model,param_grid=params,scoring=scoring,n_jobs=self.n_jobs,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(max_depth= gsearch.best_params_['max_depth'])
        print('best max_depth for rf:{}'.format(gsearch.best_params_['max_depth']))

        self.cv_score()
        self.train_score()
  

        params = {'min_samples_split':tunned_min_samples_split}
        gsearch = GridSearchCV(estimator = self.model,param_grid = params,scoring=scoring,n_jobs=self.n_jobs,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(min_samples_split = gsearch.best_params_['min_samples_split'])
        print('best min_samples_split for rf:{}'.format(gsearch.best_params_['min_samples_split']))
        self.cv_score()
        self.train_score()

        params = {'min_samples_leaf':tunned_min_samples_leaf}
        gsearch = GridSearchCV(estimator = self.model,param_grid = params,scoring=scoring,n_jobs=self.n_jobs,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(min_samples_leaf = gsearch.best_params_['min_samples_leaf'])
        print('best min_samples_leaf for rf:{}'.format(gsearch.best_params_['min_samples_leaf']))
        self.cv_score()
        self.train_score()

        params = {'max_features':tunned_max_features}
        gsearch = GridSearchCV(estimator = self.model,param_grid = params,scoring=scoring,n_jobs=self.n_jobs,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(max_features = gsearch.best_params_['max_features'])
        print('best max_features for rf:{}'.format(gsearch.best_params_['max_features']))
        self.cv_score()
        self.train_score()

        self.plot_save('RandomForestRegression')
    
        return self.model

