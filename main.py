#!/usr/bin/python

from HMM import HMM
from viterbi import viterbi
import nltk
from nltk.corpus import brown
import sys, os
import pickle

TRAIN_NUM=10000
ENABLE_RATE=False
TRAIN_NUM_RATE=0.5
MODEL_PATH = './model/hmm'

def trainModel() :
    end=TRAIN_NUM
    if ENABLE_RATE:
        end=int(TRAIN_NUM*TRAIN_NUM_RATE)
    dataSet = brown.tagged_words(tagset='universal')[:end]
    dataSet = [ [d[0].lower(), d[1]] for d in dataSet ]
    hmm = HMM(args=dataSet)
    paras = hmm.output_to_viterbi()
    # cache model
    fo = open(MODEL_PATH, 'wb')
    with fo :
        pickle.dump(paras, fo)
    return paras

def loadModel() :
    fi = open(MODEL_PATH, 'rb')
    with fi :
        paras = pickle.load(fi)
    return paras

def predictCategory(hmm, stc) :
    # preprocess: split and lower sentence
    words = nltk.word_tokenize(stc)
    words = [ word.lower() for word in words ]
    # use DP to max observed prob
    pos = viterbi(*paras, stc=words)
    print('HMM tagged:\n', pos)
    # use nltk
    taggedWords = nltk.pos_tag(words, tagset='universal')
    print('NLTK tagged:\n', taggedWords)

if __name__=='__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '-h' :
        print('HMM categorying words\n\
Usage: ./main.py [size_of_training_set] sentence_to_predict')
        exit()
    elif len(sys.argv) == 2 and sys.argv[1] == '-t' :
        '''
        使用长度为TRAIN_NUM的brown词表中的一部分为训练集，另一部分为测试集。
        注：这样很容易出现找不到对应词的错误。
        目前训练长度比例为0.5
        '''
        ENABLE_RATE=True
        paras = trainModel()
        input_words=[tuple[0] for tuple in brown.tagged_words(tagset='universal')[int(TRAIN_NUM*TRAIN_NUM_RATE):TRAIN_NUM]]
        sentence=' '.join(input_words)
        predictCategory(paras, sentence)
    elif len(sys.argv) == 2 and os.path.exists(MODEL_PATH):
        # If training num is not given, and cache exists
        # Then load model
        paras = loadModel()
    else :
        TRAIN_NUM=int(sys.argv[1])
        paras = trainModel()
    predictCategory(paras, sys.argv[-1])

from HMM import HMM
from nltk.corpus import brown
import sys


TRAIN_NUM=1000
#TRAIN_NUM_START=0
#TRAIN_NUM_END=1000


def transfer(document):
    textx_lower=[word for word in document.lower().split()]
    return 0

if __name__=='__main__':
    for i in range(1,len(sys.argv)):
        if len(sys.argv)>=2:
            TRAIN_NUM=int(sys.argv[1])
    HMM_test=HMM(brown.tagged_words()[0:TRAIN_NUM])
