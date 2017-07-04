# coding:utf-8
# 重构脚本
# 给定输入训练数据和要训练的模型名，返回调参之后的模型。
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from hyperopt import fmin,hp,tpe,STATUS_OK,Trials,space_eval
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso,Ridge
from sklearn.svm import SVR
import pickle
import time
from sklearn.metrics import r2_score
import log_class
from methods_config import param_gdr_space,param_lasso_space,param_rf_space,param_ridge_space,param_svr_space,param_xgb_space
import argparse


def load_data(train_name,test_name):
    upper_dir = '../input/'
    train_name = upper_dir+train_name
    test_name = upper_dir+test_name
    train_df = pd.read_csv(train_name)
    test_df = pd.read_csv(test_name)
    #train_df = pd.read_csv('../input/train_pca.csv')
    #test_df = pd.read_csv('../input/test_pca.csv')
    train_y = train_df['y']
    train_X = train_df.drop('y',axis=1)
    return train_X.values,train_y.values,test_df.values

class hyperopt_opt():
    model_name = ""
    best_model = None
    save_model = False
    gen_pred = False
    def __init__(self,model_name,train_X,train_y,test_X,save_model=False,gen_pred=False):
        self.model_name = model_name
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.log = log_class.log_class(model_name)
        self.save_model = save_model
        self.gen_pred = gen_pred

    def construct_model(self,param,ifbest_params=0):
        model_name = self.model_name
        if model_name == "randomforest":
            int_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
            for item in int_params:
                param[item] = int(param[item])
            self.log.add(param, 1)
            model = RandomForestRegressor(
                n_estimators=param['n_estimators'],
                max_depth=param['max_depth'],
                min_samples_split=param['min_samples_split'],
                min_samples_leaf=param['min_samples_leaf'])
        elif model_name == 'gbregressor':
            int_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
            for item in int_params:
                param[item] = int(param[item])
            self.log.add(param, 1)
            model = GradientBoostingRegressor(
                learning_rate=param['learning_rate'],
                n_estimators=param['n_estimators'],
                max_depth=param['max_depth'],
                subsample=param['subsample'],
                min_samples_split=param['min_samples_split'],
                min_samples_leaf=param['min_samples_leaf'])
        elif model_name == 'xgbregressor':
            int_params = ['max_depth', 'num_round']
            for item in int_params:
                param[item] = int(param[item])
            self.log.add(param, 1)
            model = XGBRegressor(
                n_estimators=param['num_round'],
                objective=param_xgb_space['objective'],
                learning_rate=param['eta'],
                gamma=param['gamma'],
                min_child_weight=param['min_child_weight'],
                max_depth=param['max_depth'],
                subsample=param['subsample'],
                colsample_bytree=param['colsample_bytree'],
                seed=param_xgb_space['seed'],
                nthread=param_xgb_space['nthread'])
        elif model_name == 'lasso':
            model = Lasso(alpha=param['alpha'], random_state=param_lasso_space['random_state'])
        elif model_name == 'ridge':
            model = Ridge(alpha=param['alpha'], random_state=param_ridge_space['random_state'])
        elif model_name == 'svr':
            if ifbest_params==0:
                model = SVR(C=param['C'], gamma=param['gamma'], degree=param['degree'], epsilon=param['epsilon'],
                        kernel=param['kernel'])
            else:
                if(param['kernel'] == 0):
                    cur_kernel = 'rbf'
                else:
                    assert param['kernel']==1
                    cur_kernel = 'poly'
                model = SVR(C=param['C'], gamma=param['gamma'], degree=param['degree'], epsilon=param['epsilon'],kernel = cur_kernel)
        return model

    def hyperopt_skl_obj(self,param):
        model_name = self.model_name
        train_X = self.train_X
        train_y = self.train_y
        kf = KFold(n_splits = 5)
        errors = []
        r2 = []
     
        model = self.construct_model(param)
        # to enhance the robust of local cv score ,we should repeat this cv process many times
        for train_ind,test_ind in kf.split(train_X):
            train_valid_x,train_valid_y = train_X[train_ind],train_y[train_ind]
            test_valid_x,test_valid_y = train_X[test_ind],train_y[test_ind]
            model.fit(train_valid_x,train_valid_y)
            pred_test = model.predict(test_valid_x)
            errors.append(mean_squared_error(test_valid_y,pred_test))
            r2.append(r2_score(test_valid_y,pred_test))
        model.fit(train_X,train_y)
        pred_train = model.predict(train_X)
        self.log.add('training score:'+str(r2_score(train_y,pred_train)))
        print('r2 score :')
        print(np.mean(r2))
        self.log.add('cv score of current parameter: '+str(np.mean(r2)))
        return {'loss':np.mean(errors),'status': STATUS_OK}
            
    # use the sklearn version of xgboost
    #def hyperopt_obj(self,param):
    #    # 5-fold crossvalidation error
    #    #ret = xgb.cv(param,dtrain,num_boost_round=param['num_round'])
    #    train_X = self.train_X
    #    train_y = self.train_y
    #    kf = KFold(n_splits = 3)
    #    errors = []
    #    r2 = []
    #    int_params = ['max_depth','num_round']
    #    for item in int_params:
    #        param[item] = int(param[item])
    #    self.log.add(param,1)
    #    for train_ind,test_ind in kf.split(train_X):
    #        train_valid_x,train_valid_y = train_X[train_ind],train_y[train_ind]
    #        test_valid_x,test_valid_y = train_X[test_ind],train_y[test_ind]
    #        dtrain = xgb.DMatrix(train_valid_x,label = train_valid_y)
    #        dtest = xgb.DMatrix(test_valid_x)
    #        pred_model = xgb.train(param,dtrain,num_boost_round=int(param['num_round']))
    #        pred_test = pred_model.predict(dtest)
    #        errors.append(mean_squared_error(test_valid_y,pred_test))
    #        r2.append(r2_score(test_valid_y,pred_test))
    #        
    #    pred_model.fit(train_X,train_y)
    #    pred_train = model.predict(train_X)

    #    self.log.add('training score:'+str(r2_score(train_y,pred_train)))
    #    self.log.add('cv score of current parameter: '+str(np.mean(r2)))
    #    print(np.mean(r2))
    #    print('\n')
    #    return {'loss':np.mean(errors),'status': STATUS_OK}
            
        
        
    
    def generate_prediction(self,model):
        model_name = self.model_name
        test_X = self.test_X
        if model_name == 'xgb':
            dtest = xgb.DMatrix(test_X)
            pred_test = model.predict(dtest)
            sample_df = pd.read_csv('../input/sample_submission.csv')
            test_id = sample_df['ID']
            sub_df = pd.DataFrame({'ID':test_id,'y':pred_test})
            cur_time = time.strftime("%Y-%m-%d-%H-%M",time.localtime())
            sub_df.to_csv('../submissions/hyperopt_xgb_'+cur_time+'.csv',index=False)
        else:
            pred_test = model.predict(test_X)
            sample_df = pd.read_csv('../input/sample_submission.csv')
            test_id = sample_df['ID']
            sub_df = pd.DataFrame({'ID':test_id,'y':pred_test})
            cur_time = time.strftime("%Y-%m-%d-%H-%M",time.localtime())
            sub_df.to_csv('../submissions/hyperopt_skl_'+model_name+'_'+cur_time+'.csv',index=False)  
            
    def main_tunning(self):
        model_name = self.model_name
        train_X = self.train_X
        train_y = self.train_y
        test_X = self.test_X
    
        trials = Trials()
        if model_name=='xgb':
            obj = lambda p:self.hyperopt_obj(p)
            best_params = fmin(obj,param_xgb_space,algo=tpe.suggest,max_evals=param_xgb_space['max_evals'],trials = trials)
            self.log.add(best_params,1)
            int_params = ['max_depth','num_round']
            for item in int_params:
                best_params[item] = int(best_params[item])
            
            #test the cv score  of this best_params 
            print('cv score of best parameter: ')
            self.log.add("cv score of best parameters:")

            self.hyperopt_obj(best_params)
            dtrain = xgb.DMatrix(train_X,label= train_y)
        
            trained_model = xgb.train(best_params,dtrain,num_boost_round=int(best_params['num_round']))
            cur_time = time.strftime("%Y-%m-%d-%H-%M",time.localtime())
            save2model = '../models/xgb_'+cur_time
            pickle.dump(trained_model,open(save2model,'wb'))
            self.generate_prediction(trained_model)
        #elif model_name ==
        else:

            int_params = []
            obj = lambda p:self.hyperopt_skl_obj(p)
            if model_name == 'randomforest':
                cur_param_space = param_rf_space
                int_params = ['n_estimators','max_depth','min_samples_split','min_samples_leaf']
            elif model_name =='gbregressor':
                cur_param_space = param_gdr_space
                int_params = ['n_estimators','max_depth','min_samples_split','min_samples_leaf']
            elif model_name =='xgbregressor':
                cur_param_space = param_xgb_space
                int_params = ['max_depth','num_round']
            elif model_name == 'lasso':
                cur_param_space = param_lasso_space
            elif model_name =='ridge':
                cur_param_space = param_ridge_space
            elif model_name == 'svr':
                cur_param_space = param_svr_space
                int_params = ['degree']

            best_params = fmin(obj,cur_param_space,algo=tpe.suggest,max_evals=param_rf_space['max_evals'],trials=trials)
            for item in int_params:
                best_params[item] = int(best_params[item])
            print('cv score of best paramster: ')
            self.log.add("cv score of best parameters:")
            self.hyperopt_skl_obj(best_params)

            '''if model_name == 'randomforest':
                model = RandomForestRegressor(
                            n_estimators = best_params['n_estimators'],
                            max_depth = best_params['max_depth'],
                            min_samples_split=best_params['min_samples_split'],
                            min_samples_leaf = best_params['min_samples_leaf']) 
            elif model_name == 'gbregressor':
                model = GradientBoostingRegressor(
                            learning_rate = best_params['learning_rate'],
                            n_estimators = best_params['n_estimators'],
                            subsample = best_params['subsample'],
                            max_depth = best_params['max_depth'],
                            min_samples_split = best_params['min_samples_split'],
                            min_samples_leaf = best_params['min_samples_leaf'])
            elif model_name == 'xgbregressor':
                model = XGBRegressor(
                          n_estimators = best_params['num_round'],
                          objective = param_xgb_space['objective'],
                          learning_rate = best_params['eta'],
                          gamma = best_params['gamma'],
                          min_child_weight = best_params['min_child_weight'],
                          max_depth = best_params['max_depth'],
                          subsample = best_params['subsample'],
                          colsample_bytree = best_params['colsample_bytree'],
                          seed = param_xgb_space['seed'],
                          nthread= param_xgb_space['nthread']) 
            elif model_name == 'lasso':
                model = Lasso(alpha = best_params['alpha'],random_state = param_lasso_space['random_state'])
            elif model_name == 'ridge':
                model = Ridge(alpha = best_params['alpha'],random_state = param_ridge_space['random_state'])
            elif model_name =='svr':
                kernel_ind = best_params['kernel']
                if kernel_ind == 0:
                    cur_kernel = "rbf"
                else:
                    cur_kernel = 'poly'
                model = SVR(C = best_params['C'],gamma = best_params['gamma'],degree = best_params['degree'],epsilon = best_params['epsilon'],kernel = cur_kernel)
            '''
            model =self.construct_model(best_params,1)
            model.fit(train_X,train_y)
            if self.save_model:
                cur_time = time.strftime("%Y-%m-%d-%H-%M",time.localtime())
                save2model = '../models/'+model_name+'_'+cur_time
                pickle.dump(model,open(save2model,'wb'))
            if self.gen_pred:
                self.generate_prediction(model)
            return model
    
    

def config_parser():
    parser =argparse.ArgumentParser()
    parser.add_argument('-train_name',action = 'store',dest = 'train_name',help='enter the training dataname',type=str)
    parser.add_argument('-test_name',action = 'store',dest='test_name',help = 'enter the testing dataname',type = str)
    parser.add_argument('-model_name',action = 'store',dest = 'model_name',type=str)
    parser.add_argument('-save_model',action = 'store_true',dest='save_model',default=False)
    parser.add_argument('-not_save_model',action='store_false',dest='save_model',default=False)
    parser.add_argument('-gen_pred',action='store_true',dest='genp',default=False)
    configs = parser.parse_args()
    return configs


if __name__=="__main__":
    #用传参的形式给定要训练的模型名
    #1. train_data_name,test_data_name
    #2. model_name
    #3. save_model_dir
    #4. if generate prediction for test_data and save it to csv file
    #5. pass the predictions.csv dir if above boolean flag is true

    config = config_parser()
    print(config)
    train_X,train_y,test_X = load_data(config.train_name,config.test_name)
    #methods = ['xgb','randomforest','gbregressor','lasso','xgbregressor','ridge','svr']
    #model_name = methods[5]
    xgb_opt = hyperopt_opt(config.model_name,train_X,train_y,test_X,config.save_model,config.genp)
    xgb_opt.main_tunning()
