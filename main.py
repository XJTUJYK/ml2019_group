#!/usr/bin/python

from HMM import HMM
from viterbi import viterbi
import nltk
from nltk.corpus import brown
import sys, os
import pickle

TRAIN_NUM=1000
MODEL_PATH = './model/hmm'

def trainModel() :
    dataSet = brown.tagged_words(tagset='universal')[:TRAIN_NUM]
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
    else :
        TRAIN_NUM=int(sys.argv[1])
        paras = trainModel()
    predictCategory(paras, sys.argv[-1])
