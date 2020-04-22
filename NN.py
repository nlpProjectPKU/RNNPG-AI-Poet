# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:52:47 2020

@author: zinccat
"""

import pandas as pd #用于数据处理
import numpy as np #用于矩阵计算
import torch #用于搭建及训练模型
import time #用于训练计时
import random #用于生成随机数
import os #用于文件操作
from torchtext import data #用于生成数据集
from torchtext.vocab import Vectors #用于载入预训练词向量
from tqdm import tqdm #用于绘制进度条
from torchtext.data import Iterator, BucketIterator #用于生成训练和测试所用的迭代器
import torch.nn as nn #用于搭建模型
import torch.optim as optim #用于生成优化函数
from matplotlib import pyplot as plt #用于绘制误差函数

torch.manual_seed(19260817) #设定随机数种子
torch.backends.cudnn.deterministic = True #保证可复现性
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tokenize(x): return x.split() #分词函数,后续操作中会用到

TEXT = data.Field(sequential=True, tokenize=tokenize, fix_length=7) #设定句长为50
LABEL = data.Field(sequential=False, use_vocab=False)

# 定义Dataset类
class Dataset(data.Dataset):
    name = 'Dataset'
    def __init__(self, path, text_field, label_field):
        fields = [("text", text_field), ("category", label_field)]
        examples = []
        csv_data = pd.read_csv(path) #从csv文件中读取数据
        print('read data from {}'.format(path))
        for text, label in tqdm(zip(csv_data['text'], csv_data['category'])):
            examples.append(data.Example.fromlist([str(text), label], fields))
        super(Dataset, self).__init__(examples, fields) #生成标准dataset

dataset_id = 2 #选择所使用的dataset组合
train_path = 'dataset/Train'+str(dataset_id)+'UTF8.csv' #训练数据文件路径
test_path = 'dataset/Test'+str(dataset_id)+'UTF8.csv' #测试数据文件路径
train = Dataset(train_path, text_field=TEXT, label_field=LABEL) #生成训练集
test = Dataset(test_path, text_field=TEXT, label_field=LABEL) #生成测试集

if not os.path.exists('.vector_cache'): #建立缓存文件夹以存储缓存文件
    os.mkdir('.vector_cache')
vectors = Vectors(name='weibo') #使用微博数据集所训练好的词向量
TEXT.build_vocab(train, vectors=vectors, unk_init = torch.Tensor.normal_, min_freq=5) #构建映射,设定最低词频为5
weight_matrix = TEXT.vocab.vectors #构建权重矩阵
weight_matrix.to(device)

batch_size = 1024
train_iter = BucketIterator(dataset=train, batch_size=batch_size, shuffle=True)
test_iter = Iterator(dataset=test, batch_size=batch_size, shuffle=True)

#输入: 第i句诗(7*vocab_size)输出: vi(1*embedding dim)
class CSM(nn.Module):
    def __init__(self, vocab_size = len(TEXT.vocab), pad_idx = TEXT.vocab.stoi[TEXT.pad_token], embedding_dim=300, text_len=7, feature_size=200):
        super().__init__() #调用nn.Module的构造函数进行初始化
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx) #使用embedding table构建语句到向量的映射
        self.embedding.weight.data.copy_(weight_matrix) #载入由预训练词向量生成的权重矩阵
        self.embedding.to(device)
        self.relu=nn.ReLU() #ReLU函数
        self.bn=nn.BatchNorm1d(num_features=feature_size)
        self.conv1=nn.Conv1d(in_channels=embedding_dim, out_channels=feature_size, kernel_size=2)
        self.conv2=nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=2)
        self.conv3=nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=3)
        self.conv4=nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=3)
    def forward(self, text): #前向传播
        embedded = self.embedding(text) #
        #print(text)
        embedded = embedded.permute(1, 2, 0)#batch_size*embedding_dim*text_len(7)
        out = self.conv1(embedded)
        out = self.bn(out)
        out = self.relu(out) #batch_size*feature_size*6
        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu(out) #batch_size*feature_size*5
        out = self.conv3(out)
        out = self.bn(out)
        out = self.relu(out) #batch_size*feature_size*3
        out = self.conv4(out)
        out = self.relu(out) #batch_size*feature_size*1
        #out = out.squeeze()
        return out #batch_size*feature_size

#输入: vec_i 输出: u_i^j
class RCMUnit(nn.Module):
    def __init__(self, feature_size=200):
        super().__init__()
        self.relu=nn.ReLU() #ReLU函数
        self.U = nn.Linear(in_features = feature_size, out_features = feature_size)
    def forward(self, vec): #前向传播
        out = self.U(torch.transpose(vec,0,1))
        out = self.relu(out)
        return out
        
#输入: vec_1-vec_i, 输出: u_i^1-u_i^m组成的list
class RCM(nn.Module):
    def __init__(self, feature_size=200, num_of_unit=7):
        super().__init__()
        self.relu = nn.ReLU() #ReLU函数
        self.M = nn.Linear(in_features = 2*feature_size, out_features = feature_size)
        self.U = []
        self.num_of_unit = num_of_unit
        self.feature_size=feature_size
        for i in range(0,num_of_unit-1):
            self.U.append(RCMUnit().cuda())
    def forward(self, vecs, ith_sentence): #前向传播
        print(vecs.size())
        ans = []
        h = torch.zeros((vecs.size()[0], self.feature_size)).cuda()
        for i in range(0,ith_sentence):
            out = torch.cat((vecs[:,:,i],h) ,dim=1)
            out = self.M(out)
            h = self.relu(out)
        for j in range(0, self.num_of_unit-1):
            out = self.U[j](torch.transpose(h,0,1))
            ans.append(out)
        return ans

def one_hot(x, n_class, dtype=torch.float32): 
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res

def to_onehotO(X, n_class):  
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

def to_onehot(c):
    res = torch.zeros((len(TEXT.vocab),1))
    res[TEXT.vocab.stoi[c]]=1
    return res

#输入: u_i^j,w_j 输出: 最可能的第j+1个字
class RGM(nn.Module):
    def __init__(self, vocab_size = len(TEXT.vocab), feature_size=200, text_len=7):
        super(RGM, self).__init__()
        self.vocab_size = vocab_size
        self.R = nn.Linear(feature_size, feature_size)
        self.H = nn.Linear(feature_size, feature_size)
        self.X = nn.Linear(vocab_size, feature_size)
        self.Y = nn.Linear(feature_size, vocab_size)
        self.r = torch.zeros((feature_size,1))
        self.relu = nn.ReLU()

    def forward(self, u, w, r): # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        e = to_onehot(w).cuda() # X是个list
        ro = self.R(torch.transpose(r,0,1))
        xo = self.X(torch.transpose(e,0,1))
        ho = self.H(u)
        self.r= self.relu(ro+xo+ho)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        y = self.Y(self.r)
        self.r = torch.transpose(self.r,0,1)
        return y, self.r
    
class Model(nn.Module):
    def __init__(self, vocab_size = len(TEXT.vocab), feature_size=200, text_len=7):
        super(Model, self).__init__()
        self.csm = CSM()
        self.rcm = RCM()
        self.rgm = RGM()
    def forward(self, text, ith_sentence):
        vecs = self.csm(text)
        u = self.rcm(vecs, ith_sentence)
        t = torch.zeros((200,1), requires_grad=True).cuda()
        w = '猫咪'
        length = u[-1].size()
        length = length[0]
        ans = torch.tensor([], requires_grad=True)
        for i in range(length):
            out = torch.tensor([], requires_grad=True)
            lst = []
            lst.append(w)
            for j in range(6):
                y, t = self.rgm(u[j][i].cuda(), w, t)
                #print(torch.argmax(y,dim=1))
                w = TEXT.vocab.itos[torch.argmax(y,dim=1)]
                out = torch.cat((out,TEXT.vocab.vectors[torch.argmax(y,dim=1)]), dim=1)
                lst.append(w)
            ans = torch.cat((ans,out),dim=0)
        return ans, lst

def put(str, ith_sentence):
    s = tokenize(str)
    l=[]
    print(s)
    for w in s:
        l.append([TEXT.vocab.stoi[w]])
    lt = torch.tensor(l)
    return model(lt.cuda(), ith_sentence)

def fit(epoch):
    for i in range(epoch):
        for batch in train_iter:
            #print(batch.text.size())
            model.zero_grad() #将上次计算得到的梯度值清零
            model.train() #将模型设为训练模式
            predicted, wordlist = model(batch.text.cuda(),1)
            length = batch.text.size()
            length = length[1]
            #print(length)
            comp = torch.zeros(0, requires_grad=True)
            loss = 0
            for i in range(1,7):
                comp = torch.cat((comp,TEXT.vocab.vectors[batch.text[i,:]]), dim=1)
            loss = loss_function(predicted, comp)
            loss.backward(retain_graph=True) #反向传播
            optimizer.step() #修正模型
            print(loss)
            print(wordlist)
            
model = Model()
#loss_function = nn.functional.cross_entropy #使用交叉熵损失函数
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01) #使用Adam作为优化器
model.cuda()
loss_function = nn.functional.mse_loss #使用交叉熵损失函数
#put("也 无 风 雨 也 无 晴")
#put("不要 搞个 大 新闻 呃 谔")
            
fit(1)