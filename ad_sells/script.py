#_*_coding:utf-8_*_
import numpy as np
import pandas as pd
import lightgbm as lgb
import log_class

logger = log_class.log_class('lightgbm')
train_path = './input/train.csv' 
test_path = './input/test.csv'
train_df = pd.read_csv(train_path,parse_dates=['activation_date'])
test_df = pd.read_csv(test_path,parse_dates = ['activation_date'])
train_df = train_df.sort_values(by='activation_date')

# 在这里就划分 training 和 validaiton 的index
# 将数据分成七份
num_train = train_df.shape[0]
import math
each_length = math.floor(num_train/7)
train_index = []
valid_index=  []
for i in range(5):
    train_ind = range(i*each_length,(i+2)*each_length)
    valid_ind = range((i+2)*each_length,(i+3)*each_length)
    train_index.append(train_ind)
    valid_index.append(valid_ind)


all_df = pd.concat([train_df,test_df],axis=0)
# 先处理缺失值,全都用null填充
for item in all_df.columns:
    if all_df[item].isnull().sum()!=0 and item!='price':
        all_df[item].fillna('null',inplace=True)

# 对price 取对数
# 填充log1p的均值
import numpy as np
all_df['price'][all_df['price'].notnull()] = np.log1p(all_df['price'][all_df['price'].notnull()])
all_df['price'].fillna(np.mean(all_df['price'][all_df['price'].notnull()]),inplace=True)
# all_df['price'].fillna(np.mean(np.log1p(all_df['price'][all_df['price'].notnull()])),inplace=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
russ_stop = stopwords.words('russian')
# 转换为小写
def preprocesser_lower(text):
    return text.lower()

tfidf_descrtption = TfidfVectorizer(stop_words = russ_stop,
                                    ngram_range = (1,2),
                                    preprocessor=preprocesser_lower,
                                    max_features=1000)

desc_mat = tfidf_descrtption.fit_transform(all_df['description'])
tar_label = 'deal_probability'
train_df = all_df.iloc[0:num_train,:]
test_df = all_df.iloc[num_train:,:]
# 把label 单独拿出来
train_label = train_df[tar_label]
#现在的训练特征就是tfidf_mat
train_data  = desc_mat[0:num_train,:]

import lightgbm as lgb
import sklearn.metrics  as metrics
#train_index
#valid_index
#train_label
def scores_cv_params(params):
    train_scores=[]
    valid_scores=[]
    for train_ind,valid_ind in zip(train_index,valid_index):
        train_x = train_data[train_ind,:]
        train_y = train_label[list(train_ind)]
        valid_x = train_data[valid_ind,:]
        valid_y = train_label[list(valid_ind)]
        df_train = lgb.Dataset(train_x,label=train_y)
        lmodel = lgb.train(params,df_train)
        valid_pred = lmodel.predict(valid_x)
        valid_scores.append(metrics.mean_squared_error(valid_y,valid_pred))
    return np.mean(valid_scores)

# 设置不同的参数
tunned_num_leaves = [25,31]
tunned_max_depth = [-1,20]
params={
    'num_leaves':31,
    'max_depth':-1, 
    'learning_rate':0.1,
    'n_rounds':100,
    'subsample_for_bin':200000, 
    'objective':'regression',
    'class_weight':None,
    'min_split_gain':0.0,
    'min_child_weight':0.001,
    'min_child_samples':20,
    'subsample':1.0, 
    'subsample_freq':1,
    'colsample_bytree':1.0,
    'reg_alpha':0.0, 
    'reg_lambda':0.0,
    'random_state':None,
    'n_jobs':-1, 
    'silent':True
}


cv_scores = []

for num_leaves in tunned_num_leaves:
    params['num_leaves']  = num_leaves
    cur_score = scores_cv_params(params)
    print(cur_score)
    print('\n')
    logger.add(params,ifdict=1)
    logger.add(str(cur_score))
    cv_scores.append(cur_score)
#find best num_leaves
cv_scores = np.array(cv_scores)
best_num_leaves_ind = np.argmin(cv_scores)
params['num_leaves']= tunned_num_leaves[best_num_leaves_ind]


cv_scores = []
for max_depth in tunned_max_depth:
    params['max_depth'] = max_depth
    cur_score=scores_cv_params(params)
    logger.add(params,ifdict=1)
    logger.add(str(cur_score))
    cv_scores.append(cur_score)
cv_scores = np.array(cv_scores)
best_max_depth_ind = np.argmin(cv_scores)
params['max_depth']= tunned_max_depth[best_max_depth_ind]

