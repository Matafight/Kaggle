#_*_coding:utf-8_*_

import pandas as pd
import kagglemethods.xgboost_classification as xcv




## 定义参数
## 分类任务
## 仅针对xgboost方法的参数
metric = 'log_loss'
metric_proba = True
metric_name = 'logloss'
scoring = 'neg_log_loss'
n_jobs = 4
save_model = False
scale_pos_weight = 1
if_classification = True



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
def base_model(x,y):
    xgb = xcv.xgboostClassification_CV(x,y,metric= metric,metric_proba=metric_proba,metric_name=metric_name,scoring=scoring,n_jobs=n_jobs,
            save_model=save_model,scale_pos_weight=scale_pos_weight,if_classification=if_classification)
    xgb.cross_validation()

if __name__=='__main__':
    path = './input/train.csv'
    df,feat_columns,label_column = load_data(path)
    x = df[feat_columns].values
    y = df[label_column].values
    base_model(x,y)







