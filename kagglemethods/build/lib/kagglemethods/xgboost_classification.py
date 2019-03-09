#_*_ coding:utf-8

import xgboost as xgb
import numpy as np
from xgboost.sklearn import  XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import metrics
import time
from . import xgboost_base


class xgboostClassification_CV(xgboost_base.xgboost_CV):
    """
    xgboost 用于分类任务的交叉验证代码
    """

    def __init__(self,x,y,tunning_params,metric,metric_proba=0,metric_name='auc',scoring='roc_auc',n_jobs=4,save_model=0,processed_data_version_dir= './',scale_pos_weight=1,if_classification = True):
        """
        初始化相关参数

        args:
            x: numpy array
            y: numpy array
            tunning_params: 字典类型，key是待调整参数名,values是候选集合
            metric: sklearn 中的函数，用来在交叉验证中评估验证集上的效果，不过auc 不行，因为auc的参数 不是 (y_true,y_pred) 的形式
                 optional: http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
            metric_proba: False, 表示 metric 函数是否接受模型输出0-1之间的概率值
            metric_name: 'auc' 这个参数表示调用xgboost.cv函数时用来度量的损失函数
                options :
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
            scoring: 用sklearn自带的 GridSearchCV 时需要的评估函数, 一般是越大越好。默认为 auc
                    可选项: 'neg_log_loss' 'roc_auc' ,'neg_mean_squared_error' 等
            n_jobs: 多少个线程,默认为4
            save_model: True or False, 表示是否保存模型,保存路径为 processed_data_version_dir/modules/
            processed_data_version_dir: 存放log 或者保存模型的目录,默认为 ./
            scale_pos_weight: 默认为1 对于正负样本比例不一致的数据，通过这个来控制正样本的权重
            if_classification: 是否为分类任务，默认为1，主要区别在于分类任务应该采用 StratifiedKFold做交叉验证

        """
        super(xgboostClassification_CV,self).__init__(x,y,tunning_params,metric,metric_proba = metric_proba,metric_name= metric_name,scoring=scoring,n_jobs=n_jobs,save_model=save_model,processed_data_version_dir = processed_data_version_dir,if_classification = if_classification)

        #计算负正样本的比例
        self.model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            nthread=n_jobs,
            )

    def cross_validation(self):
        bst_model = super(xgboostClassification_CV,self).cross_validation()
        self.plot_save('xgboostClassification')
        return bst_model


















