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


