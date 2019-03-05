from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
from . import learning_methods
tunned_alpha = [0.1,0.3,0.5,1,1.2,1.5,2]

class ridge_CV(learning_methods.learning_methods):
    def __init__(self,x,y,metric,metric_proba = False,scoring = 'neg_mean_squared_error',n_jobs=2,save_model=False,processed_data_version_dir='./'):
        """
        初始化相关参数

        args:
            x: numpy array
            y: numpy array
            metric: sklearn 中的函数，用来在交叉验证中评估验证集上的效果，不过auc 不行，因为auc的参数 不是 (y_true,y_pred) 的形式
                 optional: http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
            metric_proba: False, 表示 metric 函数是否接受模型输出0-1之间的概率值
            labels: 默认为[0,1]， 在分类任务中，进行预测时可能所有预测数据集都是正类或都是负类，这个参数是用来告诉metric 应该是又两个类
            scoring: 用sklearn自带的 GridSearchCV 时需要的评估函数, 一般是越大越好。默认为 neg_mean_squared_error
                    可选项: 'neg_log_loss' 'roc_auc' ,'neg_mean_squared_error' 等
            n_jobs: 多少个线程,默认为2
            save_model: True or False, 表示是否保存模型,保存路径为 processed_data_version_dir/modules/
            processed_data_version_dir: 存放log 或者保存模型的目录,默认为 ./ 
        """
        super(ridge_CV,self).__init__(x,y,metric,metric_proba=metric_proba,scoring=scoring,save_model=save_model,processed_data_version_dir=processed_data_version_dir)
        #normalize 
        self.model = Ridge(alpha=1,normalize=True)
        self.n_jobs=n_jobs



    #scoring : neg_mean_squared_error 
    def cross_validation(self):
        scoring = self.scoring
        self.train_score()
        self.cv_score()
        params = {'alpha':tunned_alpha}
        gsearch = GridSearchCV(estimator=self.model,param_grid=params,scoring=scoring,n_jobs=self.n_jobs,iid=False,cv=3)
        gsearch.fit(self.x,self.y)
        self.model.set_params(alpha = gsearch.best_params_['alpha'])
        print('best alpha for ridge:{}'.format(gsearch.best_params_['alpha']))
        self.cv_score()
        self.train_score()
        self.plot_save(name='ridge')
        return self.model

if __name__ == '__main__':
    x= None
    y = None
    metrics = None
    obj = ridge_cv(x,y,metrics)
        
       
    