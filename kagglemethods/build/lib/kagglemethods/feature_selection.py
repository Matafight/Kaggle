
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


class feature_selection():
    def __init__(self,df,feat_columns,label_column):
        self.x = df[feat_columns].values
        self.y = df[label_column].values
    
    def treebased(self):
        clf = ExtraTreesClassifier()
        clf = clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        support_ind = model.get_support()
        
        
        
        
        