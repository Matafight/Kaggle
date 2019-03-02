import xgboost as xgb
import pandas as pd
from . import log_class
from sklearn import metrics
import numpy as np
import copy
from sklearn.feature_selection  import VarianceThreshold



'''
相同的数据，相同的训练集验证集划分方式
xgboost.cv
使用方法：
1. 先用param_search 寻找最优参数
2. get_cv_score 获得最优参数的验证集效果
3. predict_test,predict_test之前需要把模型训练好
'''

class xgboost_crossvalidation():
    def __init__(self,x,y,valid_metric,processed_data_version_dir='./',folds=None,task='classification'):
        '''
        valid_metric: 观察验证集效果的度量方式，是sklearn.metrics 函数
        交叉验证过程中也是用 valid_metric 这个度量来做early_stopping
        '''
        import os
        if not os.path.exists(processed_data_version_dir):
            os.mkdir(processed_data_version_dir)
        self.x = x
        self.y = y
        self.logger = log_class.log_class('xgboost',top_level=processed_data_version_dir)
        self.folds=folds
        self.task=task
        if task=='classification':
            self.obj = 'binary:logistic'
        elif task =='regression':
            self.obj = 'reg:linear'
        else:
            raise NameError('not support such task!')
        self.best_params = {
                'objective':self.obj,
                'nthreads':4,
                'silent':1
                }
        self.valid_metric = valid_metric





    def util_search(self,dtrain,base_params,tunned_params,tunned_param_name):
        cv_score= []
        for item_para in tunned_params:
            print('tunning {}:{}'.format(tunned_param_name,str(item_para)))
            base_params[tunned_param_name]=item_para
            cvret = xgb.cv(base_params,dtrain,num_boost_round=50,nfold=3,folds=self.folds,early_stopping_rounds=5,seed=2018,shuffle=True,verbose_eval=10)
            best_round = cvret.shape[0]
            best_cv_score = cvret.iloc[best_round-1,0]
            cv_score.append(best_cv_score)
        best_ind = cv_score.index(min(cv_score))
        base_params[tunned_param_name]=tunned_params[best_ind]
        self.logger.add('best {} is {} ,cv_score: {}'.format(tunned_param_name,tunned_params[best_ind],str(min(cv_score))))
        return base_params

    def param_search(self):

        dtrain = xgb.DMatrix(self.x,label=self.y)

        # 对每个参数都有一个cv_score的list，遍历完之后argmin就可以了
        # 先按顺序把所有参数列出来
        base_params = {
                'objective':self.obj,
                'nthreads':4,
                'silent':1
                }
        # learning_rate
        tunned_learning_rate = [0.1]
        #tunned_learning_rate = [0.01,0.015,0.025,0.05,0.1]
        base_params = self.util_search(dtrain,base_params,tunned_learning_rate,'learning_rate')
        print(base_params)

        #max_depth
        tunned_max_depth = [3,5,7,9]
        base_params = self.util_search(dtrain,base_params,tunned_max_depth,'max_depth')
        print(base_params)


        ## gamma
        tunned_gamma = [0.05,0.1,0.3,0.5,0.7,0.9,1]
        base_params = self.util_search(dtrain,base_params,tunned_gamma,'gamma')
        print(base_params)

        ## min_child_weight
        #tunned_min_child_weight = [1,3,5,7]
        #base_params = self.util_seach(dtrain,base_params,tunned_min_child_weight,'min_child_weight')
        #print(base_params)

        ## colsample_bytree
        #tunned_colsample_bytree = [0.6,0.7,0.8,1]
        #base_params = util_seach(dtrain,base_params,tunned_colsample_bytree,'colsample_bytree')
        #print(base_params)

        self.best_params=base_params
        self.logger.add(base_params,ifdict=1)
        return base_params

    def selftrain(self,set_params = None):
        if set_params ==  None:
            params = self.best_params
        else:
            params = set_params
        print(params)
        dtrain = xgb.DMatrix(self.x,label = self.y)
        # early stopping 必须要搭配validatin set使用
        xgbmodel = xgb.train(params,dtrain,num_boost_round=50,verbose_eval=10)
        return xgbmodel

    def predict_test(self,test_X,test_y,thres,set_params = None):
        print('===============================training model================================')
        model = self.selftrain(set_params)
        print('===============================training finished================================')
        dtest = xgb.DMatrix(test_X,label = test_y)
        pred = model.predict(dtest)
        #split by 0.5
        pos_ind = pred>=thres
        neg_ind = pred<thres
        pred[pos_ind] = 1
        pred[neg_ind] = 0
        #miss_rule = list(np.where(test_y-pred>0)[0])
        #augmented_rule = list(np.where(pred-test_y>0)[0])
        recall = metrics.recall_score(test_y,pred)
        precision = metrics.precision_score(test_y,pred)
        accuracy = metrics.accuracy_score(test_y,pred)
        print('test recall:'+str(recall))
        print('test precision:'+str(precision))
        print('test accuracy:'+str(accuracy))
        self.logger.add("recall:{}".format(str(recall)))
        self.logger.add("prec:{}".format(str(precision)))
        self.logger.add("acc:{}".format(str(accuracy)))
        c= metrics.confusion_matrix(test_y,pred)
        if c.shape[0] >= 2:
            cm = copy.deepcopy(c)
            cm[0,0] = int(c[1,1])
            cm[0,1] = int(c[1,0])
            cm[1,0] = int(c[0,1])
            cm[1,1] = int(c[0,0])
            print(cm)
            self.logger.add("confusion matrix:{},{},{},{}".format(str(cm[0,0]),str(c[0,1]),str(c[1,0]),str(cm[1,1])))

    def evalerror(self,preds, dtrain):
        labels = dtrain.get_label()
        score = self.valid_metric(labels,preds)
        return 'eval-score-eror', score

    def get_cv_score(self,set_params = None):
        '''
        利用网络搜索到最优参数后(也可以自己设置参数,因为通常情况下不可能每次在验证集上验证时都跑一遍交叉验证，所以应该支持自己设置参数)，
        如果新加入一个新的特征，该如何迅速评估该特征是否有帮助？
        可以采用相同的参数，相同的crossvalidation 的folds设置计算cv_score
        '''
        if set_params == None:
            params = self.best_params
        else:
            params = set_params
        x=self.x
        y=self.y
        folds=self.folds
        cv_scores = []

        for (tind,vind) in folds:
            xtrain = x[tind]
            ytrain = y[tind]
            xvalid = x[vind]
            yvalid = y[vind]
            dtrain = xgb.DMatrix(xtrain,label=ytrain)
            dvalid = xgb.DMatrix(xvalid,label=yvalid)
            xgbmodel = xgb.train(params,dtrain,num_boost_round=20,evals=[(dvalid,'validation')],feval = self.evalerror,maximize = True,early_stopping_rounds=5,verbose_eval=10)
            # 返回的xgbmodel并不是最优num_boost_round的模型
            pred = xgbmodel.predict(dvalid)
            cv_scores.append(self.valid_metric(yvalid,pred))
            #cv_scores.append(metrics.log_loss(yvalid,pred))
        self.logger.add('cv score sequence:{}'.format(' '.join([str(s) for s in cv_scores])))
        self.logger.add('mean cv score:{}'.format(str(np.mean(cv_scores))))
            #保存混淆矩阵





class feat_selection():
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def treeBased(self):
        pass
    def varianceBased(self):
        sel = VarianceThreshold(threshold = 0.8*(1-0.8))
        self.x = sel.fit_transform(self.x)

    def l1Based(self):
        pass


if __name__ == '__main__':
    # 导入数据
    df_train,df_test,feat_columns,label,ident_col=load_data()
    x = df_train[feat_columns].values
    y = df_train[label].values

    #调参
    param_search(x,y)



