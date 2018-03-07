import time
import os



class log_class():
    model_path= ''
    def __init__(self,model_name):
        if not os.path.exists('./log'):
            os.mkdir('./log')
        cur_time = time.strftime("%Y-%m-%d-%H-%M",time.localtime())
        path = './log/'+model_name+'_'+cur_time+'.txt'
        with open(path,'a') as fh:
            fh.write(cur_time+'\n')
        self.model_path=path
    def add(self,info,ifdict = 0):
        if ifdict == 0: 
            with open(self.model_path,'a') as fh:
                fh.write(info+'\n')
        else:
            with open(self.model_path,'a') as fh:
                for item in info:
                    fh.write(item+':')
                    fh.write(str(info[item])+' ')

if __name__ == '__main__':
    mylog = log_class('testmodel')
    mylog.add('hello')
    params = {'id1':1,'id2':'aaa','id3':'cc'}
    mylog.add(params,1)

