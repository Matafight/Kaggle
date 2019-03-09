#_*_coding:utf-8_*_

import pandas as pd
import xgboost_classification as xcv
from sklearn import metrics
import lightgbm_classification as lcv
from configparser import ConfigParser


def get_config(config_path):
    config = ConfigParser()
    config.read(config_path)
    return config

def get_wrapper_params(config):
    ##wrapper  参数
    wrapper_params = config.items('wrapper')
    wrapper_params = {item[0]:item[1] for item in wrapper_params}
    return wrapper_params

def get_tunning_params(config):
    ## tunning 参数
    tunning_params = config.items('tunning')
    def split_para(p):
        ps = p.split(',')
        l1 =  [float(p)  for p in ps if '.' in p]
        l2 = [int(p) for p in ps if '.' not in p]
        return l1 + l2
    tunning_params = {item[0]:split_para(item[1]) for item in tunning_params}
    return tunning_params


## lightgbm相关参数
config = get_config(config_path = './config/lgb.config')
wrapper_params = get_wrapper_params(config)
## tunning 参数
tunning_params = get_tunning_params(config)
## ligthgbm 参数
lgb_params ={
        'metric':eval(wrapper_params['metric']),
        'metric_proba':int(wrapper_params['metric_proba']),
        'metric_name':wrapper_params['metric_name'],
        'scoring':wrapper_params['scoring'],
        'n_jobs':int(wrapper_params['n_jobs']),
        'save_model':int(wrapper_params['save_model']),
        }

## xgboost 相关参数
xbg_config = get_config('./config/xgb.config')
xgb_wrapper_params = get_wrapper_params(xbg_config)
xgb_tunning_params = get_tunning_params(xbg_config)

xgb_params = {
        'metric':eval(xgb_wrapper_params['metric']),
        'metric_proba':int(xgb_wrapper_params['metric_proba']),
        'metric_name':xgb_wrapper_params['metric_name'],
        'scoring':xgb_wrapper_params['scoring'],
        'n_jobs':int(xgb_wrapper_params['n_jobs']),
        'save_model':int(xgb_wrapper_params['save_model']),
        'scale_pos_weight':int(xgb_wrapper_params['scale_pos_weight']),
        'if_classification':int(xgb_wrapper_params['if_classification'])
}


    
def load_data(path):
    """

    args: input path
    return: df, feat_columns, label_column
    """
    df = pd.read_csv(path)
    label_column = 'target'
    feat_columns = []
    for i in range(0,200):
        feat_columns.append('var_'+str(i))
    return df,feat_columns,label_column

def base_model(x,y,methods='xgb'):
    if methods=='xgb':
        xgb = xcv.xgboostClassification_CV(x,y,xgb_tunning_params,metric=xgb_params['metric'],metric_proba=xgb_params['metric_proba'],metric_name=xgb_params['metric_name'],scoring=xgb_params['scoring'],n_jobs=xgb_params['n_jobs'],save_model=xgb_params['save_model'],scale_pos_weight=xgb_params['scale_pos_weight'],if_classification=xgb_params['if_classification'])
        xgb.cross_validation()
    elif methods=='lgb':
        lgb = lcv.lightgbmClassification_CV(x,y,tunning_params,metric=lgb_params['metric'],metric_proba=lgb_params['metric_proba'],metric_name=lgb_params['metric_name'],scoring=lgb_params['scoring'],n_jobs=lgb_params['n_jobs'])
        lgb.cross_validation()

if __name__=='__main__':

    df,feat_columns,label_column = load_data(config['data']['path'])
    pos_df = df.loc[df.target==1,:]
    neg_df = df.loc[df.target==0,:]
    ndf = pd.concat([pos_df.iloc[:500,:],neg_df.iloc[:500,:]],axis=0)
    x = ndf[feat_columns].values
    y = ndf[label_column].values
    method = 'xgb'
    base_model(x,y,method)







