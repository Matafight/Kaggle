

这是我写的关于kaggle比赛的一个pipeline，主要是模型调参和集成学习，目前并没有关于如何做特征工程的模板程序。主要目录如下：

# codes

代码文件夹，包含了模型调参和集成学习
 
 1. log 文件夹，保存的是各个方法做交叉验证搜索超参数的时候各个参数的cv score 以及最优参数的cv score
 2. autoGridSearch.py
 采用暴力搜索的方式搜索最优超参数
 3. entrance.py
 类似于接口脚本，避免反复在命令行中输入参数的烦恼
 4. hyperopt_models.py
 模型调参的主脚本，包含了各个模型，把需要训练的模型名当作参数传到hyperopt_models中
 5. log_class.py
 日志类
 6. methods_config.py
 hyperopt_models.py的配置文件，配置了各个方法超参数的搜索空间
 7. stacked_model.py
 采用hyperopt_models.py生成的保存在 ../models/ 路径下的模型作为stacking第一层的模型，同时在第二层调用hyperopt_models中的方法训练stacking的第二层模型。

8. tf.py
基于tensorflow的方法

# data_analysis

这是分析数据的代码，主要做数据探索和数据可视化，分析特征与标号之间的关系等。
采用jupyter notebook来做分析

# input

数据集所在文件夹

# models

利用python中的pickle保存了最优参数的模型，以模型名和当前时间来命名。

# submissions

保存模型对于test数据的预测值，用来提交到kaggle上。
