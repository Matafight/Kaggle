#_*_coding:utf-8_*_

import pandas as pd
import kagglemethods.xgboost_classification as xcv
from sklearn import metrics
import kagglemethods.lightgbm_classification as lcv
from configparser import ConfigParser
import os


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
        model = xgb.cross_validation()
    elif methods=='lgb':
        lgb = lcv.lightgbmClassification_CV(x,y,tunning_params,metric=lgb_params['metric'],metric_proba=lgb_params['metric_proba'],metric_name=lgb_params['metric_name'],scoring=lgb_params['scoring'],n_jobs=lgb_params['n_jobs'])
        model = lgb.cross_validation()
    else:
        print('NOT SUPPORT CURRENT METHOD YET!')
    return model


def generate_submission(test_df,feat_columns,model,submission_name):
    test_x = test_df[feat_columns].values
    pred = model.predict(test_x)
    ##submission
    sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})
    sub_df["target"] =pred 
    sub_df.to_csv(submission_name, index=False)



if __name__=='__main__':

    df,feat_columns,label_column = load_data(config['data']['train_path'])
    pos_df = df.loc[df.target==1,:]
    neg_df = df.loc[df.target==0,:]
    ndf = pd.concat([pos_df.iloc[:500,:],neg_df.iloc[:500,:]],axis=0)
    x = ndf[feat_columns].values
    y = ndf[label_column].values
    method = 'xgb'
    model = base_model(x,y,method)

    print('======generating submission======')
    test_df,feat_columns,label_column = load_data(config['data']['test_path'])
    test_df = test_df.iloc[:100,:]
    if not os.path.exists('./submission/'):
        os.mkdir('./submission/')
    submission_name = './submission/'+ method+'_submission.csv'
    generate_submission(test_df,feat_columns,model,submission_name)








