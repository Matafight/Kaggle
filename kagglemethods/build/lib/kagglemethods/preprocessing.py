#_*_coding:utf-8_*_
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def missing_data(data):
   '''
   描述缺失值情况
   args: pandas
   return: numpy array
   '''
   total = data.isnull().sum()
   percent = (data.isnull().sum()/data.isnull().count()*100)
   tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
   types = []
   for col in data.columns:
       dtype = str(data[col].dtype)
       types.append(dtype)
   tt['Types'] = types
   return(np.transpose(tt))


def univariate_dist(df,column,variable_type = 'cont'):
    '''
    画出单变量的分布情况，可以针对性得选择合适的方法填充缺失值
    主要要过滤缺失值
    '''
    df = df.loc[df[column].notnull(),:]
    if variable_type == 'cont':
        sns.distplot(df[column])
    elif variable_type == 'cate':
        sns.countplot(x=column, data=df)
    plt.show()

class MethodsNotFound(Exception):
      def __init__(self,method):
            Exception.__init__ (self,method)       #调用基类的__init__进行初始化
            self.method = method


def fill_missing(df,column, method='mean'):
    '''
    填充缺失值, 对于连续值,可以填充 均值，中位数，对于离散值 可以填充众数
    也可以选择 丢弃包含缺失值的样本
    methods: mean, median, mode, drop
    '''
    try:
        if method == 'mean':
            df[column] = df[column].fillna(df[column].mean())
        elif method == 'median':
            df[column] = df[column].fillna(df[column].median())
        elif method == 'mode':
            df[column] = df[column].fillna(df[column].mode()[0])
        elif method == 'drop':
            df = df.drop(df[df.columns.isnull()].index)
        else:
            raise MethodsNotFound(method)
        return df
    except MethodsNotFound as mf:
        print('methods {} not found'.format(mf.method))


def transform_cate_feat(df,column,method='onehot'):
    """ 
    两种处理离散特征的方法: 
    1. onehot
    2. labelencoder
    """
    if method == 'onehot':
        dummies = pd.get_dummies(df[column], prefix = 'f', prefix_sep = '_')
        df = pd.concat([df, dummies], axis = 1)
    elif method == 'labelencoder':
        lbl_enc = LabelEncoder()
        lbl_enc.fit(list(df[column].values))
        df[column] = lbl_enc.transform(list(df[column].values))
    return df



if __name__ == '__main__':
    df = pd.DataFrame({'a':[1,2,3,4,5,6,np.nan],'b':['r','e','d',np.nan,'r','e','e']})
    # missing data
    #print(missing_data(df))
    
    #univariate_dist(df,'a',variable_type='cont')
    #univariate_dist(df,'b',variable_type='cate')

    df = fill_missing(df,'b',method='mode')
    #df = transform_cate_feat(df,'b','labelencoder')
    print(df)





        


