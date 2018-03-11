#_*_coding:utf-8_*_

#这是一个基类，继承该类的子类可以直接使用基类的函数也可以重写基类的函数，xgboost方法就应该重写，sklearn内嵌的方法应该可以直接使用。cv_score 应该支持引入评估准则参数
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from . import log_class



class learning_methods(object):
    #parameter labels is designed for classification tasks that the training data is highly imbalanced
    def __init__(self,x,y,metric,metric_proba = False,labels = None,scoring = 'auc',save_model=False):
        self.x = x
        self.y = y
        self.metric = metric
        self.model = None
        self.metric_proba = metric_proba
        self.labels = labels
        self.save_model = save_model
        self.scoring = scoring
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

        #save train score and cv score
        logger = log_class.log_class(name)
        str_train_score ="train score sequence "+" ".join([str(item) for item in self.train_scores])
        str_cv_score = "cv score sequence"+ " ".join([str(item) for item in self.cv_scores])
        logger.add(str_train_score)
        logger.add(str_cv_score)   
        #determine if save model
        if self.save_model:
            #save model here
            from sklearn.externals import joblib
            if not os.path.exists('./modules'):
                os.mkdir('./modules')
            joblib.dump(self.model,'./modules/'+name+"_"+cur_time+".pkl")

    def train_score(self):
        self.model.fit(self.x,self.y)
        if self.metric_proba == False:
            pred_train = self.model.predict(self.x)
        else:
            pred_train = self.model.predict_proba(self.x)[:,1]
        
        if self.labels==None:
            score = self.metric(self.y,pred_train)
        else:
            score = self.metric(self.y,pred_train,labels=self.labels)
        self.train_scores.append(score)

    def cv_score(self):
        # 5-fold crossvalidation error
        kf = KFold(n_splits = 5)
        score = []
        for train_ind,test_ind in kf.split(self.x):
            train_valid_x,train_valid_y = self.x[train_ind],self.y[train_ind]
            test_valid_x,test_valid_y = self.x[test_ind],self.y[test_ind]
            self.model.fit(train_valid_x,train_valid_y)
            if self.metric_proba == False:
                pred_test = self.model.predict(test_valid_x)
            else:
                pred_test = self.model.predict_proba(test_valid_y)[:,1]

            if self.labels == None:
                score.append(self.metric(test_valid_y,pred_test))
            else:
                score.append(self.metric(test_valid_y,pred_test,labels=self.labels))

        mean_cv_score = np.mean(score)
        self.cv_scores.append(mean_cv_score)
        print('final {} on cv {}'.format(self.metric.__name__,mean_cv_score))
    
    #scoring : neg_mean_squared_error 
    def cross_validation(self,scoring):
        pass
    

