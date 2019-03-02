'''
用来对数据做eda的代码
主要功能是画图

图1. 正负样本在各个特征上的分布差异
'''
import seaborn as sns
import matplotlib.pyplot as plt

class allPlots():
    def __init__(self, df, path):
        self.df = df
        self.path = path

    def boxPlot(self, x, y):
        boxp = sns.catplot(x=x, y=y, kind="box", data=self.df)
        fig = boxp.get_figure()
        fig.savefig(path + '/tset.png')
