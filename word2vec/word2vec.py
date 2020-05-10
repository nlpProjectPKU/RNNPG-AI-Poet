# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:21:16 2020

@author: zch65458525
"""

from gensim.models import word2vec
import os

def cut(path):
    txtLists = os.listdir(path)
    fi = open('./stopwords.txt', 'r', encoding='utf-8')
    sourceInLine=fi.readlines()
    stopwords=[]
    for line in sourceInLine:
        temp1=line.strip('\n')
        stopwords.append(temp1)
    #print(stopwords)
    fo = open('./cut.txt', 'w', encoding='utf-8')
    print(txtLists)
    for txt in txtLists:
        fi = open(path+txt, 'r', encoding='utf-8')
        
        text = fi.read()  # 获取文本内容
        '''for line in text:
            for item in line:
                if item not in stopwords:
                    fo = open('./cut.txt', 'a+', encoding='utf-8')
                    fo.write(item+' ')'''#可加停用词
        fo = open('./cut.txt', 'a+', encoding='utf-8')
        fo.write(text)
        fi.close()
        fo.close()#word2vec
cut('./poemlm\\')
max_window_size = 3
k = 5 # number of negative sampling
lr = 0.001 # learning rate
num_epoch = 25
embed_size = 256
print("cut finished")
sentence=word2vec.LineSentence('cut.txt')
model =word2vec.Word2Vec(hs=1,min_count=1,window=max_window_size,
                          size=embed_size,sg=1,negative=k,min_alpha=lr);
model.build_vocab(sentence)
model.train(sentence, total_examples=model.corpus_count, epochs=num_epoch)
model.wv.save_word2vec_format('word2vec.vector')#训练完成的模型
