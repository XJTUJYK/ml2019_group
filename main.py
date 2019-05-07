#!/usr/bin/python

from HMM import HMM
from viterbi import viterbi
import nltk
from nltk.corpus import brown
import sys, os
import pickle

TRAIN_NUM=1000
ENABLE_RATE=False
TRAIN_NUM_RATE=0.5
MODEL_PATH = './model/hmm'

def trainModel() :
    if ENABLE_RATE:
        end=TRAIN_NUM*TRAIN_NUM_RATE
    else:
        end=TRAIN_NUM
    dataSet = brown.tagged_words(tagset='universal')[:end]
    dataSet = [ [d[0].lower(), d[1]] for d in dataSet ]
    hmm = HMM(dataSet)
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
    words = nltk.word_tokenize(stc)
    words = [ word.lower() for word in words ]
    pos = viterbi(*paras, words)
    print('HMM tagged:\n', pos)
    # use nltk
    taggedWords = nltk.pos_tag(words, tagset='universal')
    print('NLTK tagged:\n', taggedWords)

if __name__=='__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '-h' :
        print('HMM categorying words\n\
Usage: ./main.py [size_of_training_set] sentence_to_predict')
        exit()
    elif len(sys.argv) == 2 and os.path.exists(MODEL_PATH):
        paras = loadModel()
    elif len(sys.argv) == 2 and sys.argv[1] == '-t' :
        ENABLE_RATE=True
        paras = trainModel()
        input_words=[tuple[0] for tuple in brown.tagged_words(tagset='universal')[TRAIN_NUM*TRAIN_NUM_RATE:TRAIN_NUM]]
        sentence=' '.join(input_words)
        predictCategory(paras, )
    else :
        TRAIN_NUM=int(sys.argv[1])
        paras = trainModel()
    predictCategory(paras, sys.argv[-1])
