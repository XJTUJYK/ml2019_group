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

        self.ob_para_list=[tuple[0] for tuple in self.data]
        self.ob_para=list(set(self.ob_para_list))

        self.train()
        #self.print_para()


    def train(self):
        '''
        calculate A and B
        '''
        A, B, Pi = {}, {}, {}    # Use dict for faster index
        for i,pair in enumerate(self.data[:-1]) :
            if pair[1] in A.keys() :
                if self.data[i+1][1] in A[pair[1]].keys() :
                    A[pair[1]][self.data[i+1][1]] += 1
                else :
                    A[pair[1]][self.data[i+1][1]] = 1
            else :
                A[pair[1]] = {}
                A[pair[1]][self.data[i-1][1]] = 1
        if self.data[-1][1] not in A.keys() :
            A[self.data[-1][1]] = {}
        self.hidden_para = list(A.keys())
        self.A = [ [ A[j].get(i, 0) for i in self.hidden_para ] for j in self.hidden_para ]
        
        for pair in self.data :
            if pair[1] in B.keys() :
                if pair[0] in B[pair[1]].keys() :
                    B[pair[1]][pair[0]] += 1
                else :
                    B[pair[1]][pair[0]] = 1
                # also update Pi
                Pi[pair[1]] += 1
            else :
                B[pair[1]] = {}
                B[pair[1]][pair[0]] = 1
                Pi[pair[1]] = 1
        self.B = [ [ B[j].get(i, 0) for i in self.ob_para ] for j in self.hidden_para ]
        self.times = [ Pi[i] for i in self.hidden_para ]
        
        self.pi = [i/len(self.data) for i in self.times]
        
        self.hidden_len=len(self.hidden_para)
        self.ob_len=len(self.ob_para)
        print('Hidden states:'+str(self.hidden_len))
        print('Observed states:'+str(self.ob_len))

        self.A = [[ele/self.times[i] for ele in self.A[i]] for i in range(self.hidden_len)]
        self.B = [[ele/self.times[i] for ele in self.B[i]] for i in range(self.hidden_len)]
        
        self.end_time=time.time()
        print('Current program cost '+str(self.end_time-self.start_time)+'s')

    def print_para(self):
        self.print_pi()
        self.print_A()
        self.print_B()
        print('Current program cost '+str(self.end_time-self.start_time)+'s')

    def output_to_viterbi(self):
        '''
        返回HMM五元组
        '''
        return self.pi, self.A, self.B, self.hidden_para, self.ob_para
