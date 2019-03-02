
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tqdm
class EDA():
    '''
    共分为四大类：
        特征与特征之间
        特征与label之间
        label的分布情况
        特征的分布情况
    实际上就是 单变量分析和双变量分析。
    '''
    def __init__(self,df,feat_columns,task_type='classification',path_to_save='.'):
        '''
        输入参数：
            df,
            feat_columns，特征列
            task_type, 任务类型，回归任务:regression, 分类任务： classification
            path_to_save, 保存结果的路径
        '''
        self.df = df
        if 'label' not in df.columns:
            raise RuntimeError('input dataframe without label column')
        self.feat_columns = feat_columns
        self.task_type = task_type
        self.path_to_save = path_to_save

        if not os.path.isdir(path_to_save):
            os.mkdir(path_to_save)

        #应该基本上都是数值特征


    def label_dist(self):
        if not os.path.isdir(self.path_to_save + '/label_dist_plot'):
            os.mkdir(self.path_to_save + '/label_dist_plot')
        if self.task_type == 'classification':
            sns.countplot(x='label', data=self.df)
        elif self.task_type == 'regression':
            sns.distplot(self.df['label'])
        else:
            raise RuntimeError('unknown task type!')
        plt.savefig(self.path_to_save+'/label_dist_plot'+'/label_dist.jpg',dpi=400)

    def univariate_dist(self):
        '''
        单变量分布情况,把他们的图拼成9x9的大图
        :return:  None
        '''
        if not os.path.isdir(self.path_to_save + '/univariate_dist_plot'):
            os.mkdir(self.path_to_save + '/univariate_dist_plot')

        cnt_plots = int(len(self.feat_columns)/9)

        for i_plots in range(0,cnt_plots):
            columns = self.feat_columns[i_plots*9:(i_plots+1)*9]
            fig, ax = plt.subplots(3,3)
            # 调整每个子图的大小
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            for i in range(3):
                for j in range(3):
                    column = columns[3*i+j]
                    sns.distplot(self.df[column],ax=ax[i,j])
            plt.savefig(self.path_to_save+'/univariate_dist_plot'+'/univariate_dist_'+str(i_plots)+'.jpg',dpi=400)
        remain_feat_ind = cnt_plots*9
        remain_feat = self.feat_columns[remain_feat_ind:]
        fig, ax = plt.subplots(3, 3)
        # 调整每个子图的大小
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for i in range(3):
            for j in range(3):
                if 3*i+j < len(remain_feat):
                    column = columns[3 * i + j]
                    sns.distplot(self.df[column], ax=ax[i, j])
        plt.savefig(self.path_to_save+'/univariate_dist_plot'+'/univariate_dist_' +str(cnt_plots)+'.jpg',dpi=400)

    def boxplot_withlabel(self):
        '''
        对于分类任务 就是连续变量与离散变量之间的关系plot
        '''

        if not os.path.isdir(self.path_to_save + '/boxplot_feature_label_plot'):
            os.mkdir(self.path_to_save + '/boxplot_feature_label_plot')

        cnt_plots = int(len(self.feat_columns) / 9)
        for i_plots in range(0, cnt_plots):
            columns = self.feat_columns[i_plots * 9:(i_plots + 1) * 9]
            fig, ax = plt.subplots(3, 3)
            fig.subplots_adjust(hspace=0.6, wspace=0.4)
            for i in range(3):
                for j in range(3):
                    column = columns[3 * i + j]
                    b = sns.boxplot(x='label', y=column,data=self.df,ax=ax[i,j])
                    b.set_ylabel(column,fontsize=5)
            plt.savefig(self.path_to_save +'/boxplot_feature_label_plot' +'/boxplot_with_label_' + str(i_plots) + '.jpg', dpi=400)

        remain_feat_ind = cnt_plots * 9
        remain_feat = self.feat_columns[remain_feat_ind:]
        fig, ax = plt.subplots(3, 3)
        fig.subplots_adjust(hspace=0.6, wspace=0.4)
        for i in range(3):
            for j in range(3):
                if 3 * i + j < len(remain_feat):
                    column = columns[3 * i + j]
                    sns.boxplot(x='label', y=column,data=self.df,ax=ax[i,j])
        plt.savefig(self.path_to_save + '/boxplot_feature_label_plot'+'/boxplot_with_label_' + str(cnt_plots) + '.jpg', dpi=400)

    def pair_plot(self):

        # compute all the pairwise features w.r.t label
        if not os.path.isdir(self.path_to_save+'/pairwise_scatter_plot'):
            os.mkdir(self.path_to_save +'/pairwise_scatter_plot')

        max_pic  = 10000
        cnt = 0
        feat_columns = self.feat_columns
        for i in range(len(feat_columns)):
            for j in range(i+1,len(feat_columns)):
                if cnt<=max_pic:
                    sub_set=[feat_columns[i],feat_columns[j],'label']
                    df_tar = self.df[sub_set]
                    g = sns.pairplot(df_tar,hue = 'label')
                    plt.savefig(self.path_to_save+'/pairwise_scatter_plot/'+'scatter_'+str(cnt)+'.jpg',dpi=400)
                    cnt +=1













