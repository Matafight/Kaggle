#_*_coding:utf-8_*_
import sys
sys.path.append('..')
#sys.path.append('../kaggle_methods')
from dm_methods.kaggle_methods.ridge import ridge_cv
from dm_methods.kaggle_methods.xgboost_classification import xgboost_classification_cv
from dm_methods.kaggle_methods.logistic_regression import LogisticRegression_CV
#from ridge import ridge_cv
from sklearn import metrics


import time

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

#处理数据
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


df_train = pd.read_csv('./input/train.csv')
df_test = pd.read_csv('./input/test.csv')

df_train = df_train.loc[:1000,:]
df_test = df_test.loc[:1000,:]


label_name = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
train_label = df_train[label_name]
df_all = pd.concat([df_train[['id','comment_text']],df_test],axis=0)


import re
from nltk.corpus import stopwords
import string
eng_words = stopwords.words('english')
def word_tokenize(x):
    regex = re.compile('['+re.escape(string.punctuation)+'0-9\\n\\t]')
    text = regex.sub(' ',x)
    words = [word for word in text.split(' ') if len(word)>=1]
    words = [word.lower() for word in words if word not in eng_words]
    return words

from sklearn.feature_extraction.text import TfidfVectorizer
comment_texts = df_all['comment_text'].apply(word_tokenize)
comment_texts = [" ".join(text) for text in comment_texts]
df_all['comment_processed'] = comment_texts
tfidf = TfidfVectorizer()
tfidf_vector = tfidf.fit_transform(df_all['comment_processed'])

from sklearn.decomposition import TruncatedSVD
pca = TruncatedSVD(n_components= 100)
pca.fit(tfidf_vector)
pca_transformed = pca.transform(tfidf_vector)

training = pca_transformed[:df_train.shape[0],:]
testing = pca_transformed[df_train.shape[0]:,:]

'''submissions = pd.DataFrame({'id':df_test['id']})
metric = metrics.mean_squared_error
scoring = 'neg_mean_squared_error'
for cur_label in label_name:
    label = train_label[cur_label]
    with Timer() as t:
        ridge_cls = ridge_cv(training,label,metric)
        ridge_model = ridge_cls.cross_validation(scoring =scoring) 
        prediction = ridge_model.predict(testing)
        submissions[cur_label] = prediction
    print('time interval for one class is {}'.format(t.interval))

submissions.to_csv('submission_ridge.csv',index=False)'''

'''submissions = pd.DataFrame({'id':df_test['id']})
metric = metrics.log_loss
scoring = 'neg_log_loss'
for cur_label in label_name:
    label = train_label[cur_label]
    with Timer() as t:
        xgb_cls = xgboost_classification_cv(training,label,metric)
        xgb_model = xgb_cls.cross_validation(scoring =scoring) 
        prediction =xgb_model.predict(testing)
        submissions[cur_label] = prediction
    print('time interval for one class is {}'.format(t.interval))

submissions.to_csv('submission_xgboost.csv',index=False)'''

submissions = pd.DataFrame({'id':df_test['id']})
metric = metrics.log_loss
scoring = 'neg_log_loss'
for cur_label in label_name:
    label = train_label[cur_label]
    with Timer() as t:
        lr_cls = LogisticRegression_CV(training,label,metric)
        lr_model = lr_cls.cross_validation(scoring =scoring) 
        prediction = lr_model.predict(testing)
        submissions[cur_label] = prediction
    print('time interval for one class is {}'.format(t.interval))

submissions.to_csv('submission_lr.csv',index=False)