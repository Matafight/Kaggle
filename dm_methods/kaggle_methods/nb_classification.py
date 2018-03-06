#_*_coding:utf-8_*_

#naive bayes 方法没有超参数
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB,MultinomialNB


class GaussianNB_CV():
    def __init__(self,x,y,metric):
        self.x = x
        self.y = y
        self.metric= metric
        self.model = MultinomialNB()
    def cross_validation(self,scoring = 'roc_auc'):
        #the scoring parameter is useless in naive bayes, I keep to make sure the api consistent with other methods
        self.model.fit(self.x,self.y)
        return self.model


