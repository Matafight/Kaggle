import numpy as np
import pandas as pd
import pickle
from dm_methods.kaggle_methods.logistic_regression import LogisticRegression_CV


def load_data():
    train_path = './input/train_data.dat'
    train_label_path = './input/train_label.dat'
    test_path = './input/test_data.dat'
    train_data = pickle.load(open(train_path,'rb'))
    test_data = pickle.load(open(test_path,'rb'))
    train_label= pickle.load(open(train_label_path,'rb'))
    return train_data,test_data,train_label

if __name__=='__main__':
    train_data,test_data,train_label = load_data()
    print(train_data.shape)
    print(test_data.shape)
    print(train_label.shape)


