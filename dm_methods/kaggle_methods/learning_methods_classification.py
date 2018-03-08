#_*_coding:utf-8_*_

#这是一个基类，继承该类的子类可以直接使用基类的函数也可以重写基类的函数，xgboost方法就应该重写，sklearn内嵌的方法应该可以直接使用。cv_score 应该支持引入评估准则参数
#将验证集上的结果存到log中去
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import numpy as np



#the sklearn embeded method do not need the metric_name para as in the xgboost or lightgbm models
class learning_methods(object):
    def __init__(self,x,y,metric):
        self.x = x
        self.y = y
        self.metric = metric
        self.model = None
        
        


    def cv_score(self):
        # 5-fold crossvalidation error
        kf = KFold(n_splits = 5)
        score = []
        for train_ind,test_ind in kf.split(self.x):
            train_valid_x,train_valid_y = self.x[train_ind],self.y[train_ind]
            test_valid_x,test_valid_y = self.x[test_ind],self.y[test_ind]
            self.model.fit(train_valid_x,train_valid_y)
            pred_test = self.model.predict(test_valid_x)
            try:
                score.append(self.metric(test_valid_y,pred_test))
            except Exception as e:
                print(e)
                score.append(self.metric(test_valid_y,pred_test,labels=[0,1]))
                
        ret = 'final {} on cv {}'.format(self.metric.__name__,np.mean(score))
        return ret
    def train_score(self):
        pred = self.model.predict(self.x)
        ret = 'train {} on train {} '.format(self.metric.__name__,self.metric(self.y,pred))
        return ret
    #scoring : neg_mean_squared_error 
    def cross_validation(self):
        pass
    

