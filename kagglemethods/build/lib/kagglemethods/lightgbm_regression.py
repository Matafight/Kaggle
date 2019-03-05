#_*_coding:utf-8
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from . import log_class
import matplotlib.pyplot as plt
from . import lightgbm_base

class lightgbmRegression_CV(lightgbm_base.lightgbm_CV):
    """
    lightgbm 用于回归任务的交叉验证代码
    """
    def __init__(self,x,y,metric,metric_proba=False,metric_name='l2',scoring = 'neg_mean_squared_error',n_jobs=2,save_model=False,processed_data_version_dir='./'):
        """
        初始化相关参数
        
        args:
            x: numpy array
            y: numpy array
            metric: sklearn 中的函数，用来在交叉验证中评估验证集上的效果，不过auc 不行，因为auc的参数 不是 (y_true,y_pred) 的形式
                 optional: http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
            metric_proba: False, 表示 metric 函数是否接受模型输出0-1之间的概率值
            metric_name: 'l2' 这个参数表示 调用lightgbm.cv函数时用来度量的损失函数，可选值为
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
            save_model: True or False, 表示是否保存模型,保存路径为 processed_data_version_dir/modules/
            processed_data_version_dir: 存放log 或者保存模型的目录,默认为 ./
        """
        super(lightgbmRegression_CV,self).__init__(x,y,metric,metric_proba=metric_proba,metric_name=metric_name,scoring=scoring,n_jobs=n_jobs,save_model = save_model,processed_data_version_dir = processed_data_version_dir)
        self.model = LGBMRegressor(silent=True,n_jobs=n_jobs)
   
  


    def cross_validation(self):
        bst_model = super(lightgbmRegression_CV,self).cross_validation()
        self.plot_save()
        return bst_model