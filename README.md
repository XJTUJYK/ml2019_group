项目描述
基于HMM与Vetebi的词性标注

采用python的nltk自然语言处理包，通过该包下载brown语料库

关于引入nltk的说明参考http://www.52nlp.cn/tag/%E5%B8%83%E6%9C%97%E8%AF%AD%E6%96%99%E5%BA%93

pip install nltk
import nltk
nltk.download()

在弹出的图形界面中下载all-corpora,其中含有brown，若报错，尝试命令行下载

nltk.download('all')
若报以下错，尝试手动下载，参考https://blog.csdn.net/joey_su/article/details/17289621

[nltk_data] Error loading all: <urlopen error [WinError 10060]

以下为查看brown语料库的实例代码

from nltk.corpus import brown
print(brown.readme())
print(brown.words()[0:10])
print(brown.tagged_words()[0:10])
应出现：
BROWN CORPUS

A Standard Corpus of Present-Day Edited American
English, for use with Digital Computers.

by W. N. Francis and H. Kucera (1964)
Department of Linguistics, Brown University
Providence, Rhode Island, USA

Revised 1971, Revised and Amplified 1979

http://www.hit.uib.no/icame/brown/bcm.html

Distributed with the permission of the copyright holder,
redistribution permitted.

['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', 'Friday', 'an', 'investigation', 'of']
[('The', 'AT'), ('Fulton', 'NP-TL'), ('County', 'NN-TL'), ('Grand', 'JJ-TL'), ('Jury', 'NN-TL'), ('said', 'VBD'), ('Friday', 'NR'), ('an', 'AT'), ('investigation', 'NN'), ('of', 'IN')]
