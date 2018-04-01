#_*_ coding:utf-8 _*_

#from . import log_class
import pandas as pd
import numpy as np
import os
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn import metrics
from ridge import ridge_CV
from . import ridge
class stacking(object):
    def __init__(self,x,y,test_x,module_dir='./modules',task = 'regression'):
        self.x = x
        self.y = y
        self.test_x = test_x
        #load models,获得module_dir目录下的所有文件
        moduleNames=  os.listdir(module_dir)
        self.modules = []
        self.SEED = 1024
        self.task = task
        for name in moduleNames:
            self.modules.append(joblib.load(name))
        print('initializing finished ...')

    def get_oof(self,clf):
        x = self.x
        y = self.y
        test_x = self.test_x
        NFOLDS = 5
        SEED = self.SEED
        oof_train = np.zeros((x.shape[0],))
        oof_test = np.zeros((test_x.shape[0],))
        oof_test_skf = np.empty((NFOLDS, test_x.shape[0]))
        # since kf is a generator object, and it can't be reused, so we should add this line in this function
        kf = KFold(n_splits= NFOLDS, random_state=SEED).split(x)
        for i,(train_index, test_index) in enumerate(kf):
            x_tr = x[train_index]
            y_tr = y[train_index]
            x_te = x[test_index]
            clf.fit(x_tr, y_tr)
            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(test_x)
        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    def generate_fir_layer(self):
        print('generating first layer predictions...')
        modules = self.modules
        len_modules = len(modules)
        ntrain = self.x.shape[0]
        ntest = self.test_x.shape[0]
        train_firlayer = np.zeros((ntrain,len_modules))
        test_firlayer = np.zeros((ntest,len_modules))
        i = 0
        for i,clf in enumerate(modules):
            train_sample,test_sample = self.get_oof(clf)
            train_firlayer[:,i] = train_sample.ravel()
            test_firlayer[:,i] = test_sample.ravel()
        return train_firlayer,test_firlayer


    def stack(self):
        train_fir,test_fir = self.generate_fir_layer()
        trainX = np.concatenate((self.x,train_fir),axis=1)
        testX = np.concatenate((self.test_x,test_fir),axis=1)

        #second layer module can use the existing methods
        metric = metrics.mean_squared_error
        scoring = 'neg_mean_squared_error'
        ridge_cls = ridge_CV(trainX,self.y,metric=metric,scoring=scoring)
        ridge_model = ridge_cls.cross_validation()
        pred = ridge_model.predict(testX)
        return pred

        