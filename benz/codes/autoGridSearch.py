# _*_ coding:utf-8 _*_
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn  import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
import pickle
import time


def load_data_processed():
    train_df = pd.read_csv('../input/processed_train.csv')
    test_df = pd.read_csv('../input/processed_test.csv')
    test_ori_df = pd.read_csv('../input/test.csv')
    train_y = train_df['y']
    train_X = train_df.drop('y',axis=1)
    test_ids = test_ori_df.ID.values
    return train_X,train_y,test_df,test_ids

def load_data():
    train_df = pd.read_csv('./input/train.csv')
    test_df = pd.read_csv('./input/test.csv')
    test_ids = test_df['ID']
    # get train_y, test ids and unite datasets to perform
    train_y = train_df['y']
    train_df.drop('y', axis = 1, inplace = True)
    test_ids = test_df.ID.values
    all_df = pd.concat([train_df,test_df], axis = 0)
    
    categorical =  ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]
    for f in categorical:
        dummies = pd.get_dummies(all_df[f], prefix = f, prefix_sep = '_')
        all_df = pd.concat([all_df, dummies], axis = 1)
    
    # drop original categorical features
    all_df.drop(categorical, axis = 1, inplace = True)
    # get feature dataset for test and training        
    train_X = all_df.drop(["ID"], axis=1).iloc[:len(train_df),:]
    test_X = all_df.drop(["ID"], axis=1).iloc[len(train_df):,:]
    return train_X,train_y,test_X,test_ids

def modelfit(alg, dtrain, ytrain , useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    print('in modelfitting...')
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain, label=ytrain)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round= 500, nfold=cv_folds,
                          metrics='rmse', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        print('num_rounds %f'%cvresult.shape[0])
    alg.fit(dtrain, ytrain,eval_metric='rmse')
    dtrain_predictions = alg.predict(dtrain)
    print ("\nR2 score on the train data:")
    print ("R2 : %.4g" % metrics.r2_score(ytrain, dtrain_predictions))


def cv_score(model,train_X,train_y):
    # 5-fold crossvalidation error
    kf = KFold(n_splits = 5)
    r2 = []
    params = model.get_xgb_params()
    print('final model parameter :')
    print(params)
    for train_ind,test_ind in kf.split(train_X):
        train_valid_x,train_valid_y = train_X[train_ind],train_y[train_ind]
        test_valid_x,test_valid_y = train_X[test_ind],train_y[test_ind]
        dtrain = xgb.DMatrix(train_valid_x,label = train_valid_y)
        dtest = xgb.DMatrix(test_valid_x)
        pred_model = xgb.train(params,dtrain,num_boost_round=int(params['n_estimators']))

        pred_test = pred_model.predict(dtest)
        r2.append(metrics.r2_score(test_valid_y,pred_test))
    print('final r2 score on cv:')
    print(np.mean(r2))

def cross_validation(dtrain,ytrain,predictors):
    #每次调整完一个参数，重新确定新的num_rounds
    dtrain = dtrain[predictors]
    xgb_model = XGBRegressor(
                learning_rate= 0.5,
                max_depth = 20,
                n_estimators = 100,
                min_child_weight = 1,
                gamma = 0,
                objective='reg:linear',
                nthread=4,
                )
    modelfit(xgb_model,dtrain,ytrain)
    print('tunning learning rate...')
    params = {'learning_rate':[0.01,0.015,0.025,0.05,0.1]}
    gsearch = GridSearchCV(estimator = xgb_model,param_grid = params, scoring = 'neg_mean_squared_error',n_jobs = 4,iid=False,cv=5)
    gsearch.fit(dtrain.values,ytrain)
    xgb_model.set_params(learning_rate = gsearch.best_params_['learning_rate'])
    print(gsearch.best_params_)

    print('tunning max_depth...')
    params = { 'max_depth':[3,5,7,9]}
    print(xgb_model.get_params()['n_estimators'])
    gsearch = GridSearchCV(estimator = xgb_model,param_grid = params, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
    gsearch.fit(dtrain.values,ytrain)
    xgb_model.set_params(max_depth = gsearch.best_params_['max_depth'])
    print(gsearch.best_params_)
    #choose best num_round
    modelfit(xgb_model,dtrain,ytrain)
    print(xgb_model.get_params()['n_estimators'])
    
    print('tunning min_child_weight...')
    param_child_weight = {'min_child_weight':[1,3,5,7]}
    gsearch = GridSearchCV(estimator = xgb_model,param_grid = param_child_weight, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
    gsearch.fit(dtrain.values,ytrain)
    xgb_model.set_params(min_child_weight = gsearch.best_params_['min_child_weight'])
    print(xgb_model.get_params())
    modelfit(xgb_model,dtrain.values,ytrain)
    print(xgb_model.get_params()['n_estimators'])

    print('tunning gamma...')
    param_gamma = {'gamma':[0.05,0.1,0.3,0.5,0.7,0.9,1]}
    gsearch = GridSearchCV(estimator = xgb_model,param_grid = param_gamma, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
    gsearch.fit(dtrain.values,ytrain)
    xgb_model.set_params(gamma = gsearch.best_params_['gamma'])
    print(xgb_model.get_params())
    modelfit(xgb_model,dtrain.values,ytrain)
    print(xgb_model.get_params()['n_estimators'])

    #print('tunning colsample_bylevel')
    #param_colsample_bylevel = {'colsample_bylevel':[0.6,0.8,1]}
    #gsearch = GridSearchCV(estimator = xgb_model,param_grid = param_colsample_bylevel, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
    #gsearch.fit(dtrain.values,ytrain)
    #xgb_model.set_params(colsample_bylevel = gsearch.best_params_['colsample_bylevel'])
    #tunning colsample_bytree
    print(xgb_model.get_params())
    modelfit(xgb_model,dtrain.values,ytrain)
    print('num_rounds after tunning colsample_bylevel:%f'%xgb_model.get_params()['n_estimators'])

    print('tunning colsample_bytree...')
    param_colsample_bytree = {'colsample_bytree':[0.6,0.7,0.8,1]}
    gsearch = GridSearchCV(estimator = xgb_model,param_grid = param_colsample_bytree, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
    gsearch.fit(dtrain.values,ytrain)
    xgb_model.set_params(colsample_bytree = gsearch.best_params_['colsample_bytree'])
    print(xgb_model.get_params())
    modelfit(xgb_model,dtrain.values,ytrain)
    print('num_rounds after tunning colsample_bytree:%f'%xgb_model.get_params()['n_estimators'])
    # save and return model
    cur_time = time.strftime("%Y-%m-%d-%H-%M",time.localtime())
    pickle.dump(xgb_model,open('../models/autogridsearch_xgb_'+cur_time+'.model','wb'))
    cv_score(xgb_model,dtrain.values,ytrain)
    return xgb_model
    

    

def generate_results(dtrain,ytrain,dtest,xgb_model,test_ids):
    xgb_model.fit(dtrain.values,ytrain)
    pred_test = xgb_model.predict(dtest.values)
    sub = pd.DataFrame({'ID':test_ids,'y':pred_test})
    cur_time = time.strftime("%Y-%m-%d-%H-%M",time.localtime())
    sub.to_csv('../submissions/autogridsearch_'+cur_time+'.csv',index=False)
    
    
if __name__ == '__main__':
    dtrain,ytrain,dtest,test_ids=load_data_processed()
    predictors = dtrain.columns
    xgb_model = cross_validation(dtrain,ytrain,predictors)
    generate_results(dtrain,ytrain,dtest,xgb_model,test_ids)

