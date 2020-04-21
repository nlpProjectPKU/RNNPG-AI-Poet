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

batch_size = 1
train_iter = BucketIterator(dataset=train, batch_size=batch_size, shuffle=True)
test_iter = Iterator(dataset=test, batch_size=batch_size, shuffle=True)

class TextCNN(nn.Module):
    def __init__(self, window_sizes, vocab_size = len(TEXT.vocab), pad_idx = TEXT.vocab.stoi[TEXT.pad_token], embedding_dim=300, text_len=50, output_dim=9, feature_size=100):
        super().__init__() #调用nn.Module的构造函数进行初始化
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx) #使用embedding table构建语句到向量的映射
        self.embedding.weight.data.copy_(weight_matrix) #载入由预训练词向量生成的权重矩阵
        self.convs = nn.ModuleList([ #定义所使用的卷积操作
                nn.Sequential(nn.Conv1d(in_channels=embedding_dim, out_channels=feature_size, kernel_size=h,), #1维卷积
                              nn.BatchNorm1d(num_features=feature_size),  #正则化
                              nn.ReLU(), #ReLU
                              nn.MaxPool1d(kernel_size=text_len-h+1)) #Max Pooling
                              for h in window_sizes])
        self.fc1 = nn.Linear(in_features=feature_size*len(window_sizes),out_features=50) #全连接层
        self.dropout = nn.Dropout(0.2) #dropout
        self.fc2 = nn.Linear(in_features=50,out_features=9) #全连接层
        
    def forward(self, text): #前向传播
        embedded = self.embedding(text)
        embedded = embedded.permute(1, 2, 0) #[]
        out = [conv(embedded) for conv in self.convs]
        out = torch.cat(out, dim=1) #纵向拼接卷积操作输出的矩阵
        out = out.view(-1, out.size(1)) #将矩阵拉直为向量
        out = self.fc1(out)
        out = self.dropout(out)
        y = self.fc2(out) #通过全连接层处理获得预测类别
        return y #返回预测值

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
        print(text)
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
        out = out.squeeze()
        return out #batch_size*feature_size

#输入: vec_i 输出: 
class RCMUnit(nn.Module):
    def __init__(self, feature_size=200):
        super().__init__()
        self.relu=nn.ReLU() #ReLU函数
        self.U = nn.Linear(in_features = feature_size, out_features = feature_size)
    def forward(self, vec): #前向传播
        out = self.U(torch.transpose(vec,0,1))
        out = self.relu(out)
        return out
        
class RCM(nn.Module):
    def __init__(self, feature_size=200, num_of_unit=7):
        super().__init__()
        self.relu = nn.ReLU() #ReLU函数
        self.M = nn.Linear(in_features = 2*feature_size, out_features = feature_size)
        self.h0 = torch.zeros((feature_size,1))
        self.U = []
        self.num_of_unit = num_of_unit
        for i in range(0,num_of_unit-1):
            self.U.append(RCMUnit())
    def forward(self, vecs, ith_sentence): #前向传播
        print(vecs.size())
        ans = []
        h = self.h0
        for i in range(0,ith_sentence-1):
            out = torch.cat((vecs[i],h) ,dim=0)
            out = self.M(out)
            h = self.relu(out)
        for j in range(0, self.num_of_unit-1):
            out = self.U[j](h)
            ans.append(out)
        return ans

class RCM(nn.Module):
    def __init__(self, feature_size=200, num_of_unit=7):
        super().__init__()
        self.relu = nn.ReLU() #ReLU函数
        self.M = nn.Linear(in_features = 2*feature_size, out_features = feature_size)
        self.h0 = torch.zeros((feature_size,1))
        self.U = []
        self.num_of_unit = num_of_unit
        for i in range(0,num_of_unit-1):
            self.U.append(RCMUnit())
    def forward(self, vecs, ith_sentence): #前向传播
        print(vecs.size())
        ans = []
        h = self.h0
        for i in range(0,ith_sentence-1):
            out = torch.cat((vecs[i],h) ,dim=0)
            out = self.M(out)
            h = self.relu(out)
        for j in range(0, self.num_of_unit-1):
            out = self.U[j](h)
            ans.append(out)
        return ans
    
'''
def put(str):
    s = tokenize(str)
    l=[]
    print(s)
    for w in s:
        l.append([TEXT.vocab.stoi[w]])
    lt = torch.tensor(l)
    return model(lt.cuda())
'''
model1 = CSM() #定义TextCNN模型
model2 = RCM()
loss_function = nn.functional.cross_entropy #使用交叉熵损失函数
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model1.parameters()), lr=0.001) #使用Adam作为优化器
model1.cuda() #将模型移至gpu
model2.cuda()
#put("也 无 风 雨 也 无 晴")
#put("不要 搞个 大 新闻 呃 谔")
i=1
for batch in train_iter:
    predicted = model1(batch.text.cuda())
    predicted = model2(predicted,1)
    print(predicted[1].size())
    i+=1
    if i>=2:
        break