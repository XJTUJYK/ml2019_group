import numpy as np


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
        self.data=args
        self.hidden_para_list=[tuple[1] for tuple in self.data]
        self.hidden_para=list(set(self.hidden_para_list))#去除重复的隐藏状态元素
        self.hidden_len=len(self.hidden_para)

        self.ob_para_list=[tuple[0] for tuple in self.data]
        self.ob_para=list(set(self.ob_para_list))
        self.ob_len=len(self.ob_para)
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
        for i in range(self.hidden_len):
            for j in range(self.hidden_len):
                sum_j_after_i=0#对于两词性上下文相连出现的计数，相当于准备计算隐藏状态转移矩阵的元素（只有一层马尔可夫）
                for k in range(len(self.hidden_para_list)-1):
                    if self.hidden_para[i]==self.hidden_para_list[k] and self.hidden_para[j]==self.hidden_para_list[k+1]:#i刚好在j前面
                        sum_j_after_i+=1
                self.A[i][j]=sum_j_after_i/self.times[i]#i在j前出现的次数除以i出现的次数，相当于条件概率j|i，就是i到j的转移概率

        for i in range(self.hidden_len):
            for j in range(self.ob_len):
                num_j_match_i=0#对于某一词对应（匹配）某一词性的次数，相当于准备计算混淆矩阵的元素
                for k in range(len(self.data)):
                    if self.data[k][0]==self.ob_para[j] and self.data[k][1]==self.hidden_para[i]:#输入串中i刚好是词j所对应的词性
                        num_j_match_i+=1
                self.B[i][j]=num_j_match_i/self.times[i]#i和j匹配的次数除以词性i出现的次数，相当于条件概率j|i，就是词性i到词j的转移概率



    def print_para(self):
        self.print_pi()
        self.print_A()
        self.print_B()


    def print_pi(self):
        print(self.pi)


    def print_A(self):
        print(self.A)


    def print_B(self):
        print(self.B)


    def output_to_viterbi(self):
        return self.pi,self.A,self.B,self.hidden_para,self.ob_para