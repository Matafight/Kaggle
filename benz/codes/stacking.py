# _*_ coding:utf-8_*_
import numpy as np
from sklearn.linear_model import ElasticNetCV, ElasticNet
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso,LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_squared_error
import time
from hyperopt import fmin,hp,tpe,STATUS_OK,Trials,space_eval
from hyperopt_models import hyperopt_opt

xgb_min_num_round = 10
xgb_max_num_round = 80
xgb_num_round_step = 5
xgb_random_seed = 10
skl_random_seed = 20
param_xgb_space = {
    'task': 'regression',
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'eta': hp.quniform('eta', 0.01, 1, 0.01),
    'gamma': hp.quniform('gamma', 0, 2, 0.1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'max_depth': hp.quniform('max_depth', 1, 10, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.1),
    'num_round': hp.quniform('num_round', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
    'nthread': 5,
    'silent':1,
    'seed': xgb_random_seed,
    "max_evals": 200,
}



class ensemble():
    models=[]
    train_firlayer = None
    test_firlayer = None
    
    def __init__(self):
        self.NFOLDS = 3
        self.SEED = 10
        self.train_X,self.train_y,self.test_X = self.load_data()
        self.load_models()
        print('initialization...')

    def load_data(self):
        train_df = pd.read_csv('../input/processed_train.csv')
        test_df = pd.read_csv('../input/processed_test.csv')
        print('loading data...')
        #train_df = pd.read_csv('../input/train_pca.csv')
        #test_df = pd.read_csv('../input/test_pca.csv')
        train_y = train_df['y']
        train_X = train_df.drop('y',axis=1)
        return train_X.values,train_y.values,test_df.values

    def load_models(self):
        model_names = ['ridge_2017-06-30-14-25']
        for item in model_names:
            print('../models/'+item)
            print(type('../models/'+item).__name__)
            #fname = ()

            model = pickle.load(open('../models/'+item,'rb'))
            self.models.append(model)

    def get_oof(self,clf):
        x_train = self.train_X
        y_train = self.train_y
        x_test = self.test_X
        NFOLDS = self.NFOLDS
        SEED = self.SEED
        oof_train = np.zeros((x_train.shape[0],))
        oof_test = np.zeros((x_test.shape[0],))
        oof_test_skf = np.empty((NFOLDS, x_test.shape[0]))
        # since kf is a generator object, and it can't be reused, so we should add this line in this function
        kf = KFold(n_splits= NFOLDS, random_state=SEED).split(x_train)
        for i,(train_index, test_index) in enumerate(kf):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]
            clf.fit(x_tr, y_tr)
            oof_train[test_index] = clf.predict(x_te)
            oof_test[:] = oof_test_skf.mean(axis=0)
            oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    def generate_fir_layer(self):
        print('generating first layer predictions...')
        models = self.models
        len_models = len(models)
        ntrain = self.train_X.shape[0]
        ntest = self.test_X.shape[0]
        train_firlayer = np.zeros((ntrain,len_models))
        test_firlayer = np.zeros((ntest,len_models))
        i = 0
        for i,clf in enumerate(models):
            train_sample,test_sample = self.get_oof(clf)
            train_firlayer[:,i] = train_sample.ravel()
            test_firlayer[:,i] = test_sample.ravel()
        self.train_firlayer = train_firlayer
        self.test_firlayer = test_firlayer

    def cv_score(self,model,train_X,train_y):
        kf = KFold(n_splits = 3)
        r2 = []
        for train_ind,test_ind in kf.split(train_X):
            train_valid_x,train_valid_y = train_X[train_ind],train_y[train_ind]
            test_valid_x,test_valid_y = train_X[test_ind],train_y[test_ind]
            model.fit(train_valid_x,train_valid_y)
            pred_test = model.predict(test_valid_x)
            r2.append(r2_score(test_valid_y,pred_test))
        print('final cv score is :')
        print(np.mean(r2))

    def generate_submission(self,model,test_X):
        preds = model.predict(test_X)
        sample_df = pd.read_csv('../input/sample_submission.csv')
        test_ids = sample_df['ID']
        cur_time = time.strftime("%Y-%m-%d-%H-%M",time.localtime())
        df_sub = pd.DataFrame({'ID': test_ids, 'y': preds})
        df_sub.to_csv('../submissions/stacking_'+cur_time+'.csv',index=False)
    def hyperopt_obj(self,param,train_X,train_y):
        # 5-fold crossvalidation error
        #ret = xgb.cv(param,dtrain,num_boost_round=param['num_round'])
        kf = KFold(n_splits = 3)
        errors = []
        r2 = []
        int_params = ['max_depth','num_round']
        for item in int_params:
            param[item] = int(param[item])
        for train_ind,test_ind in kf.split(train_X):
            train_valid_x,train_valid_y = train_X[train_ind],train_y[train_ind]
            test_valid_x,test_valid_y = train_X[test_ind],train_y[test_ind]
            dtrain = xgb.DMatrix(train_valid_x,label = train_valid_y)
            dtest = xgb.DMatrix(test_valid_x)
            pred_model = xgb.train(param,dtrain,num_boost_round=int(param['num_round']))
            pred_test = pred_model.predict(dtest)
            errors.append(mean_squared_error(test_valid_y,pred_test))
            r2.append(r2_score(test_valid_y,pred_test))
        all_dtrain = xgb.DMatrix(train_X,label = train_y)
        print('training score:')
        pred_model = xgb.train(param,all_dtrain,num_boost_round= int(param['num_round']))
        all_dtest = xgb.DMatrix(train_X)
        pred_train = pred_model.predict(all_dtest)
        print(str(r2_score(train_y,pred_train)))
        print(np.mean(r2))
        print('\n')
        return {'loss':np.mean(errors),'status': STATUS_OK}

    def hyper_xgb_main(self,train_X,train_y):
         obj = lambda p:self.hyperopt_obj(p,train_X,train_y)
         trials = Trials()
         best_params = fmin(obj,param_xgb_space,algo=tpe.suggest,max_evals=param_xgb_space['max_evals'],trials = trials)
         int_params = ['max_depth','num_round']
         for item in int_params:
             best_params[item] = int(best_params[item])
         #test the cv score  of this best_params 
         print('cv score of best parameter: ')
         cur_param = param_xgb_space
         for item in best_params:
             cur_param[item] = best_params[item]
         self.hyperopt_obj(cur_param,train_X,train_y)
         dtrain = xgb.DMatrix(train_X,label= train_y)
         trained_model = xgb.train(cur_param,dtrain,num_boost_round=int(cur_param['num_round']))
         self.generate_submission(trained_model,self.test_X)
    # need to know the cv score
    # use xgboost as the last layer
    def final_lay(self):
        # we can combine the meta-training data with the output from first layer
        self.generate_fir_layer()
        train_X = np.concatenate((self.train_X,self.train_firlayer),axis =1)
        train_y = self.train_y
        test_X = np.concatenate((self.test_X,self.test_firlayer),axis = 1)
        print(train_X.shape)
        print(train_y.shape)
        #cv_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, .995, 1], eps=0.001, n_alphas=100, fit_intercept=True,
        #                        normalize=True, precompute='auto', max_iter=2000, tol=0.0001, cv=5,
        #                        copy_X=True, verbose=0, n_jobs=-1, positive=False, random_state=None, selection='cyclic')
        ## train model with best parameters from CV
        #cv_model.fit(train_X,train_y)
        #model = ElasticNet(l1_ratio=cv_model.l1_ratio_ ,alpha = cv_model.alphas_, max_iter=cv_model.n_iter_, fit_intercept=True, normalize = True)
        #model.fit(train_X, train_y)
        #print(r2_score(train_y, cv_model.predict(train_X)))
        #self.cv_score(cv_model,train_X,train_y)
        fc = hyperopt_opt('xgbregressor',train_X,train_y,train_X)
        final_model = fc.main_tunning()
        self.generate_submission(final_model,test_X)
       



if __name__ == '__main__':
    ens_model = ensemble()
    ens_model.final_lay()

