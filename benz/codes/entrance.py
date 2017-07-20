# _*_ coding:utf-8_*_
import os
os.system('python hyperopt_models.py -model_name randomforest -train_name processed_train.csv -test_name processed_test.csv -save_model -gen_pred')

os.system('python hyperopt_models.py -model_name gbregressor -train_name processed_train.csv -test_name processed_test.csv -save_model -gen_pred')
os.system('python hyperopt_models.py -model_name lasso -train_name processed_train.csv -test_name processed_test.csv -save_model -gen_pred')
