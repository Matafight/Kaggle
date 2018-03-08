#_*_coding:utf-8_*_

#这是一个基类，继承该类的子类可以直接使用基类的函数也可以重写基类的函数，xgboost方法就应该重写，sklearn内嵌的方法应该可以直接使用。cv_score 应该支持引入评估准则参数
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import numpy as np



class learning_methods(object):
    def __init__(self,x,y,metric):
        self.x = x
        self.y = y
        self.metric = metric
        self.model = None

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

    def cv_score(self):
        # 5-fold crossvalidation error
        kf = KFold(n_splits = 5)
        score = []
        for train_ind,test_ind in kf.split(self.x):
            train_valid_x,train_valid_y = self.x[train_ind],self.y[train_ind]
            test_valid_x,test_valid_y = self.x[test_ind],self.y[test_ind]
            self.model.fit(train_valid_x,train_valid_y)
            pred_test = self.model.predict(test_valid_x)
            score.append(self.metric(test_valid_y,pred_test))
        mean_cv_score = np.mean(score)
        self.cv_scores.append(mean_cv_score)
        print('final {} on cv {}'.format(self.metric.__name__,mean_cv_score))
    
    #scoring : neg_mean_squared_error 
    def cross_validation(self,scoring):
        pass
    

