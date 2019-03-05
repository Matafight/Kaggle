import time
import os



class log_class():
    """
    用来记录训练过程指标的日志类

    """
    model_path= ''
    def __init__(self,model_name,top_level='./',no_time=0):
        """
        初始化相关参数

        args:
            model_name: 模型名称，对不同模型创建不同的目录
            top_level:  日志存放目录, 默认为 './'
            no_time: 0 or 1，表示是否对日志加上logtime， 0表示加上时间
        """
        if not os.path.exists(top_level+'/log'):
            os.mkdir(top_level+'/log')
        cur_time = time.strftime("%Y-%m-%d-%H-%M",time.localtime())
        if no_time == 0:
            path = top_level+'/log/'+model_name+'_'+cur_time+'.txt'
        else:
            path = top_level+'/log/'+model_name+'.txt'

        with open(path,'a') as fh:
            fh.write(cur_time+'\n')
        self.model_path=path

    def add(self,info,ifdict = 0):
        """
        添加结果到日志txt中，可以直接存放文本，也可以存放字典

        args:
            info: 要存的信息,可以是 string ，也可以是 dict
            ifdict: 表示info是否为词典, 默认为0
            
        """
        if ifdict == 0: 
            with open(self.model_path,'a') as fh:
                fh.write(info+'\n')
        else:
            with open(self.model_path,'a') as fh:
                for item in info:
                    fh.write(item+':')
                    fh.write(str(info[item])+'\n')

if __name__ == '__main__':
    mylog = log_class('testmodel')
    mylog.add('hello')
    params = {'id1':1,'id2':'aaa','id3':'cc'}
    mylog.add(params,1)

