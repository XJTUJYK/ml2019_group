import numpy as np
import time


class HMM():    
    '''
    HMM有三个参数，pi代表隐藏状态的先验概率，A代表隐藏状态转移概率矩阵，B为一个混淆矩阵又称发射矩阵，代表从给定隐藏状态下可观测状态的概率矩阵
    首先解决给定数据集训练这三个参数的问题，brown数据集都给定了标签，实现有监督学习即可，较为简单
    '''

    def __init__(self,args):
        '''
        接收一个元组的列表，每个元组有两个元素，第一个为可观测状态数据，第二个为隐藏状态数据
        pi的key为隐藏状态，value为先验概率
        '''
        self.start_time=time.time()
        self.data=args
        self.hidden_para_list=[tuple[1] for tuple in self.data]
        self.hidden_para=list(set(self.hidden_para_list))#去除重复的隐藏状态元素
        self.hidden_len=len(self.hidden_para)

        self.ob_para_list=[tuple[0] for tuple in self.data]
        self.ob_para=list(set(self.ob_para_list))
        self.ob_len=len(self.ob_para)
        #为每个词性出现的次数单独计数，在计算矩阵时直接索引较为方便
        self.times=np.array([self.hidden_para_list.count(para) for para in list(set(self.hidden_para_list))])

        self.pi=self.times/len(self.hidden_para_list)
        self.A=np.zeros([self.hidden_len,self.hidden_len])
        self.B=np.zeros([self.hidden_len,self.ob_len])
        self.train()
        self.print_para()


    def train(self):
        '''
        calculate A and B
        '''
        print('M:'+str(self.hidden_len))
        print('N:'+str(self.ob_len))
        #直接遍历原数据，复杂度为O(logm*logm*k)
        d_copy=list(self.data).copy()
        while(len(d_copy)!=1):
            #如果词性x在词性y前面，就在矩阵[x][y]处+1，这样遍历原数据
            self.A[self.hidden_para.index(d_copy[0][1])][self.hidden_para.index(d_copy[1][1])]+=1
            d_copy.pop(0)       
        #复杂度为O(logm*logn*k)，其中n约等于k的几分之一
        d_copy=list(self.data).copy()
        while(len(d_copy)!=0):
            self.B[self.hidden_para.index(d_copy[0][1])][self.ob_para.index(d_copy[0][0])]+=1
            d_copy.pop(0)
        for i in range(self.hidden_len):
            #除以先验概率得到条件概率，虽然这里是次数/次数
            self.A[i][:]/=self.times[i]
            self.B[i][:]/=self.times[i]
        self.end_time=time.time()


    def print_para(self):
        self.print_pi()
        self.print_A()
        self.print_B()
        print('Current program cost '+str(self.end_time-self.start_time)+'s')


    def print_pi(self):
        print(self.pi)


    def print_A(self):
        print(self.A)


    def print_B(self):
        print(self.B)


    def output_to_viterbi(self):
        '''
        返回HMM五元组
        '''
        return self.pi,self.A,self.B,self.hidden_para,self.ob_para