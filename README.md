# Kaggle
This is the codes I write to participate the Kagge competition

1. [benz](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing)

This is the first competition I take it seriously, I write my complete auto hyperparameter tunning codes for this tasks,and also implement the stacking methods.

2. MNIST    
3. Titanic
# kaggle methods
## dm_methods
包含了在数据挖掘中常用的算法，将交叉验证等过程封装起来，只需要传入训练数据就可以返回交叉验证后的模型。
- xgboost 
- lightgbm
- ridge
- logistic regression
- randomforest
- naive bayes

一些特性：
- 可以画出训练和验证集的性能随着参数的变化曲线
- 交叉验证过程对用户透明，使用简单
- 所有方法具有相同的接口函数，这意味着可以通过传入函数名的字符串调用相应的函数，类似于工厂模式？
- sklearn内置的函数继承同一父类
- xgboost、lightgbm回归和分类的代码继承同一父类

todo:
- 基于上述方法构建ensemble的代码，包括 bagging,stacking,blending
