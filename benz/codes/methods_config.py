
from hyperopt import hp,tpe,STATUS_OK,Trials,space_eval
import numpy as np

xgb_min_num_round = 10
xgb_max_num_round = 80
xgb_num_round_step = 5
xgb_random_seed = 10
skl_random_seed = 20
# we should use the huber loss function
# for hp.choice function, the best_params returned by fmin is the index of the list, so try not use hp.choice
param_xgb_space = {
    'task': 'regression',
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'eta': hp.quniform('eta', 0.01, 1, 0.01),
    'gamma': hp.quniform('gamma', 0, 2, 0.1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'max_depth': hp.quniform('max_depth', 1, 10, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.1),
    'num_round': hp.quniform('num_round', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
    'nthread': 5,
    'silent':1,
    'seed': xgb_random_seed,
    "max_evals": 200,
}

param_gdr_space={
        "learning_rate":hp.quniform('learning_rate',0.01,0.5,0.01),
        "n_estimators":hp.quniform('n_estimators',xgb_min_num_round,xgb_max_num_round,xgb_num_round_step),
        'max_depth': hp.quniform('max_depth', 1, 10, 1),
        'min_samples_split':hp.quniform('min_samples_split',2,10,1),
        'subsample':hp.quniform('subsample',0.5,1,0.1),
        'min_samples_leaf':hp.quniform('min_samples_leaf',2,10,1),
        "max_evals": 200
        }

param_rf_space={
    'n_estimators': hp.quniform('n_estimators', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
    'max_depth': hp.quniform('max_depth', 1, 10, 1),
    'min_samples_split':hp.quniform('min_samples_split',2,10,1),
    'min_samples_leaf':hp.quniform('min_samples_leaf',2,10,1),
     "max_evals": 200
}


param_svr_space= {
    'C': hp.loguniform("C", np.log(1), np.log(100)),
    'gamma': hp.loguniform("gamma", np.log(0.001), np.log(0.1)),
    'degree': hp.quniform('degree', 1, 5, 1),
    'epsilon': hp.loguniform("epsilon", np.log(0.001), np.log(0.1)),
    'kernel': hp.choice('kernel', ['rbf', 'poly']),
    "max_evals":200,
}

## ridge regression
param_ridge_space = {
    'alpha': hp.loguniform("alpha", np.log(0.01), np.log(20)),
    'random_state': skl_random_seed,
    "max_evals": 200,
}

## lasso
param_lasso_space = {
    'alpha': hp.loguniform("alpha", np.log(0.00001), np.log(0.1)),
    'random_state': skl_random_seed,
    "max_evals": 200,
}

### logistic regression
#param_lr_space = {
#    'C': hp.loguniform("C", np.log(0.001), np.log(10)),
#    'random_state': skl_random_seed,
#    "max_evals": 200,
#}
