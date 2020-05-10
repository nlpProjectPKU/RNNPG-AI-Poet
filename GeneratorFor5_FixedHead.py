#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
import torch  # 用于搭建及训练模型
import torch.nn as nn  # 用于搭建模型
from torchtext import data  # 用于生成数据集
from torchtext.vocab import Vectors  # 用于载入预训练词向量
from torch.autograd import Variable
from torchtext.data import BucketIterator  # 用于生成训练和测试所用的迭代器
import os.path as path
import codecs
import numpy as np

import kenlm
import operator
import random
import copy

# In[2]:


PATH = {
    "SHIXUEHANYING_PATH": "./dataset/shixuehanying.txt",
    "TOP_RESULT": "./shengcheng/top.txt",
    "TONAL_PATH": "./dataset/pingshui.txt",
    "MODEL": "./model/model5_2.pth",
    "VECTOR": "./model/word2vec7.vector",
    "DATA": "./data/",
}


# In[3]:


def tokenize(x): return x.split()


text_len = 5
TEXT = data.Field(sequential=True, tokenize=tokenize)


class Dataset(data.Dataset):
    name = 'Dataset'

    def __init__(self, fin, text_field):
        fields = [("text", text_field)]
        examples = []
        # print('read data from {}'.format(path))
        for line in fin:
            examples.append(data.Example.fromlist([line], fields))
        super(Dataset, self).__init__(examples, fields)  # 生成标准dataset


def getDataIter(fin, fiveOrSeven):
    data = Dataset(fin, TEXT)
    vectors = Vectors(PATH["VECTOR"])
    TEXT.build_vocab(data, vectors=vectors, unk_init=torch.Tensor.normal_, min_freq=1)  # 构建映射,设定最低词频为5
    return


def getTrainIter(fiveOrSeven):
    assert fiveOrSeven == 5 or fiveOrSeven == 7
    trainfin = codecs.open(path.join(PATH["DATA"], "qtrain" + str(fiveOrSeven)), 'r', encoding='utf-8')
    return getDataIter(trainfin, fiveOrSeven)


# In[4]:


orderDict = torch.load(PATH["MODEL"], map_location='cpu')
feature_size = orderDict["conv1.weight"].size()[0]
vocab_size = orderDict["embedding.weight"].size()[0]
embedding_dim = orderDict["embedding.weight"].size()[1]

getTrainIter(text_len)
weight_matrix = TEXT.vocab.vectors  # 构建权重矩阵

# In[5]:


FIVE_PINGZE = [[[0, -1, 1, 1, -1], [0, 1, -1, -1, 1], [0, 1, 1, -1, -1], [0, -1, -1, 1, 1]],
               [[0, -1, -1, 1, 1], [0, 1, -1, -1, 1], [0, 1, 1, -1, -1], [0, -1, -1, 1, 1]],
               [[0, 1, 1, -1, -1], [0, -1, -1, 1, 1], [0, -1, 1, 1, -1], [0, 1, -1, -1, 1]],
               [[1, 1, -1, -1, 1], [0, -1, -1, 1, 1], [0, -1, 1, 1, -1], [0, 1, -1, -1, 1]]]

SEVEN_PINGZE = [[[0, 1, 0, -1, -1, 1, 1], [0, -1, 1, 1, -1, -1, 1], [0, -1, 0, 1, 1, -1, -1], [0, 1, 0, -1, -1, 1, 1]],
                [[0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, -1, -1, 1], [0, -1, 0, 1, 1, -1, -1], [0, 1, 0, -1, -1, 1, 1]],
                [[0, -1, 1, 1, -1, -1, 1], [0, 1, 0, -1, -1, 1, 1], [0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, -1, -1, 1]],
                [[0, -1, 0, 1, 1, -1, -1], [0, 0, -1, -1, -1, 1, 1], [0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, -1, -1, 1]]]


# In[6]:


def idx_to_onehot(w, vocab_size, batch_size):
    res = torch.zeros((batch_size, vocab_size)).scatter_(1, w, 1)
    return torch.transpose(res, 0, 1)


# In[7]:


class Model(nn.Module):
    def __init__(self, vocab_size, weight_matrix, pad_idx, embedding_dim=150, feature_size=200, text_len=7,
                 dropout=0.2):
        super(Model, self).__init__()
        self.feature_size = feature_size
        self.vocab_size = vocab_size
        self.text_len = text_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding.weight.data.copy_(weight_matrix)  # 载入由预训练词向量生成的权重矩阵
        self.relu = nn.LeakyReLU()  # ReLU函数
        # self.bn = nn.BatchNorm1d(num_features=feature_size)
        self.d = nn.Dropout(p=dropout)
        # CSM所用的卷积层
        self.conv1 = nn.Conv1d(in_channels=embedding_dim,
                               out_channels=feature_size, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=feature_size,
                               out_channels=feature_size, kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=feature_size,
                               out_channels=feature_size, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=feature_size,
                               out_channels=feature_size, kernel_size=3)
        self.M = nn.Linear(in_features=2 * feature_size,
                           out_features=feature_size)
        # RCM
        self.U1 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U2 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U3 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U4 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U5 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U6 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U7 = nn.Linear(in_features=feature_size, out_features=feature_size)
        # RGM
        self.R = nn.Linear(feature_size, feature_size)
        self.H = nn.Linear(feature_size, feature_size)
        self.X = nn.Linear(embedding_dim, feature_size)
        self.Y = nn.Linear(feature_size, vocab_size)

    def forward(self, text, state, ith_sentence, ith_character):  # text 28*batch_size
        # --------------------------------------------------------------------- #
        # 词嵌入
        embedded = self.embedding(text)
        embedded = embedded.permute(1, 2, 0)
        # --------------------------------------------------------------------- #
        # CSM部分
        vecs = []
        for j in range(1, ith_sentence):  # 生成v1-v_ith_sentence-1
            out = self.conv1(embedded[:, :, (j - 1) * self.text_len: j * self.text_len])
            out = self.d(out)
            out = self.relu(out)  # batch_size*feature_size*6
            out = self.conv2(out)
            out = self.d(out)
            out = self.relu(out)  # batch_size*feature_size*5
            out = self.conv3(out)
            out = self.d(out)
            out = self.relu(out)  # batch_size*feature_size*3
            if self.text_len == 7:
                out = self.conv4(out)
                out = self.d(out)
                out = self.relu(out)  # batch_size*feature_size*1
            vecs.append(out.squeeze(2))
        # --------------------------------------------------------------------- #
        # RCM部分
        h = torch.zeros((1, self.feature_size))
        for i in range(0, ith_sentence - 1):
            out = torch.cat((vecs[i], h), dim=1)
            out = self.M(out)
            h = self.relu(out)
        if ith_character == 1:
            rcmout = self.U1(h)
        elif ith_character == 2:
            rcmout = self.U2(h)
        elif ith_character == 3:
            rcmout = self.U3(h)
        elif ith_character == 4:
            rcmout = self.U4(h)
        elif ith_character == 5:
            rcmout = self.U5(h)
        elif ith_character == 6:
            rcmout = self.U6(h)
        elif ith_character == 7:
            rcmout = self.U7(h)
        rcmout = self.relu(rcmout)
        # --------------------------------------------------------------------- #
        # RGM部分
        w = text[self.text_len * (ith_sentence - 1) + ith_character - 2]
        middle = self.R(state)
        w = self.embedding(w)  # 单字
        state = self.relu(middle + self.X(w) + self.H(rcmout))
        y = self.Y(state)
        return y, state

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.feature_size))


model = Model(vocab_size=len(TEXT.vocab), weight_matrix=weight_matrix, pad_idx=TEXT.vocab.stoi[TEXT.pad_token],
              embedding_dim=embedding_dim, feature_size=feature_size, text_len=text_len)
model.load_state_dict(orderDict)  # load model


# In[8]:


def read_shixuehanying():
    # get shixuehanying
    categories = []
    labels = []
    words = []
    with open(PATH["SHIXUEHANYING_PATH"], 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            line = line.strip()
            if line:
                if line[0] == '<':
                    if line[1] == 'b':
                        titles = line.split('\t')
                        categories.append(titles[2])
                else:
                    line = line.split('\t')
                    if len(line) == 3:
                        tmp = line[2].split(' ')
                        tmp.append(line[1])
                    else:
                        tmp = line[1]
                    if len(tmp) >= 10:
                        labels.append(categories[len(categories) - 1] + "-" + line[1])
                        words.append(tmp)
            else:
                break

    all_diction = []  # all character
    fir_diction = []  # first character
    for cate in words:
        for string in cate:
            fir_diction += string[0]
            for ch in string:
                all_diction += ch
    all_diction = list(set(all_diction))
    fir_diction = list(set(fir_diction))
    return words, all_diction, fir_diction


def read_character_tone():
    # get tonal dictionary of each character
    ping = []
    ze = []
    with open(PATH["TONAL_PATH"], "r") as f:
        isPing = False
        for line in f.readlines():
            line = line.strip()
            if line:
                if line[0] == '/':
                    isPing = not isPing
                    continue
                for i in line:
                    if isPing:
                        ping.append(i)
                    else:
                        ze.append(i)
    return {"Ping": ping, "Ze": ze}


# In[9]:


def user_input():
    # get structure, subject, words
    words, all_diction, fir_diction = read_shixuehanying()
    chars = 5
    while True:
        print("Please input 4 characters for fix head: ")
        characters = input()
        if len(characters) == 4 and characters[0] in fir_diction:
            if characters[1] in all_diction and characters[2] in all_diction and characters[3] in all_diction:
                break
        print("Invalid input. Please try again.")

    fir = []
    for idx, cate in enumerate(words):
        for w in cate:
            if w[0] == characters[0]:
                fir.append(w)
    return fir, chars, characters


# In[10]:


tonal_hash = read_character_tone()


def judge_fir_tonal_pattern(row, chars):
    # remove poem with duplicated characters
    if len(row) != len(set(row)):
        return -1
    # judge rhythm availability
    tone = FIVE_PINGZE if chars == 5 else SEVEN_PINGZE
    for i in range(0, 4):
        for j in range(0, chars + 1):
            if j == chars:
                return i
            if tone[i][0][j] == 0:
                continue
            elif tone[i][0][j] == 1 and row[j] in tonal_hash["Ping"]:
                continue
            elif tone[i][0][j] == -1 and row[j] in tonal_hash["Ze"]:
                continue
            else:
                break
    return -1


# In[11]:


def judge_tonal_pattern(row):
    # decide tonal pattern according to first line
    if len(row) != len(set(row)):
        return -1
    # judge rhythm availability
    chars = len(row)
    tone = FIVE_PINGZE if chars == 5 else SEVEN_PINGZE
    for i in range(0, 4):  # each tonal pattern
        for j in range(0, chars + 1):
            if j == chars:
                return i
            if tone[i][0][j] == 0:
                continue
            elif tone[i][0][j] == 1 and row[j] in tonal_hash["Ping"]:
                continue
            elif tone[i][0][j] == -1 and row[j] in tonal_hash["Ze"]:
                continue
            else:
                break
    return -1


# In[12]:


def judge_tonal(rows):
    # input a given poem and judge tonal pattern
    chars = len(rows[0])  # [["A","A",...],["B",...],...]
    tone = FIVE_PINGZE if chars == 5 else SEVEN_PINGZE
    pattern = judge_tonal_pattern(rows[0])
    if pattern == -1:
        return -1
    for i, row in enumerate(rows):
        for j, ch in enumerate(row):
            if tone[pattern][i][j] == 1 and ch not in tonal_hash["Ping"]:
                return -1
            if tone[pattern][i][j] == -1 and ch not in tonal_hash["Ze"]:
                return -1
    return 1


# In[13]:


def id2char(num): return TEXT.vocab.itos[num]


def sentence_to_onehot(idx, chars):
    res = np.zeros((4 * chars, 1))
    for id1, sen in enumerate(idx):
        for id2, ch in enumerate(sen):
            res[id1 * chars + id2] = TEXT.vocab.stoi[ch]
    return Variable(torch.LongTensor(res))


# In[14]:


def generate_fir_sentence(topn=5, expend=3):
    fir, chars, head = user_input()
    candidate = []
    tmp = [[head[i]] for i in range(0, 4)]  # initialize
    for string in fir:
        tmp[0] = [ch for ch in string]
        candidate.append(copy.deepcopy(tmp))

    language_model = kenlm.Model("first.poem.lm")
    model.eval()

    for i in range(2, chars + 1):
        tmp = candidate[:]
        candidate = []
        for sen in tmp:
            if len(sen[0]) >= i:  # no need to produce
                candidate.append(sen)
            else:
                state = torch.zeros((1, feature_size), requires_grad=True)
                input_var = sentence_to_onehot(sen, chars)  # 20 * 1
                for k in range(1, i + 1):
                    out, state = model(input_var, state, 1, k)  # predict
                # state = torch.zeros((1, feature_size), requires_grad=True)
                # out, state = model(input_var, state, i+1, j)

                poss = out.data.reshape(-1).numpy().tolist()  # according to dl model
                get_top = []
                for _id, p in enumerate(poss):
                    get_top.append((_id, p))  # (id, possibility)
                get_top = sorted(get_top, key=lambda x: x[1], reverse=True)
                time = 0  # select top 2
                pt = 0
                while time < expend:
                    ch = id2char(get_top[pt][0])  # id to char
                    tmpflag = True
                    for each in sen:  # avoid duplicate
                        if ch in each:
                            tmpflag = False
                            break
                    if not tmpflag:
                        pt += 1
                        continue
                    sen[0].append(ch)
                    time += 1
                    pt += 1
                    candidate.append(copy.deepcopy(sen))
                    sen[0].pop()

    tmp = candidate[:]
    candidate = []
    for lines in tmp:
        if judge_fir_tonal_pattern(''.join(lines[0]), 5) >= 0:
            candidate.append(lines)
    score = []  # score after whole sentence
    for lines in candidate:
        score.append((lines, language_model.score(" ".join(lines[0]))))  # score the last sentence
    score = sorted(score, key=lambda x: x[1], reverse=True)
    score = score[0: min(topn, len(score))]
    candidate = [lines[0] for lines in score]
    return candidate


# In[15]:


def generate(topn=10, expend=3):
    candidate = generate_fir_sentence()  # [[["A"],["B"],["C"],["D"]]]
    print("generating...")
    chars = 5
    language_model = kenlm.Model("first.poem.lm")
    model.eval()

    for i in range(1, 4):  # for each line(1-4)
        for j in range(2, chars + 1):  # for each character(2-chars)
            tmp = candidate[:]
            candidate = []
            for sen in tmp:  # [["A","A",...],["B",...],...]
                state = torch.zeros((1, feature_size), requires_grad=True)
                input_var = sentence_to_onehot(sen, chars)  # 20 * 1
                for k in range(2, j + 1):
                    out, state = model(input_var, state, i + 1, k)  # predict
                # state = torch.zeros((1, feature_size), requires_grad=True)
                # out, state = model(input_var, state, i+1, j)

                poss = out.data.reshape(-1).numpy().tolist()  # according to dl model
                get_top = []
                for _id, p in enumerate(poss):
                    get_top.append((_id, p))  # (id, possibility)
                get_top = sorted(get_top, key=lambda x: x[1], reverse=True)
                time = 0  # select top 2
                pt = 0
                while time < expend:
                    ch = id2char(get_top[pt][0])  # id to char
                    tmpflag = True
                    for each in sen:  # duplicate
                        if ch in each:
                            tmpflag = False
                            break
                    if not tmpflag:
                        pt += 1
                        continue
                    sen[i].append(ch)
                    if judge_tonal(sen):  # add into candidate
                        time += 1
                        candidate.append(copy.deepcopy(sen))
                    pt += 1
                    sen[i].pop()

        score = []  # score after a whole sentence
        for lines in candidate:
            score.append((lines, language_model.score(" ".join(lines[-1]))))  # score the last sentence
        score = sorted(score, key=lambda x: x[1], reverse=True)
        score = score[0: min(topn, len(score))]
        candidate = [lines[0] for lines in score]
    return candidate


# In[16]:


def print_topn(topn=10):
    candidate = generate(topn)
    for poem in candidate:
        cnt = 0
        for sen in poem:
            cnt += 1
            if cnt % 2 == 0:
                print(''.join(sen) + '。')
            else:
                print(''.join(sen) + '，')
        print()
    return


# In[17]:


# print_topn()

# In[ ]:
