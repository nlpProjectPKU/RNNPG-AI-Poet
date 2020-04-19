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

def tokenize(x): return x.split() #分词函数,后续操作中会用到

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

class CSM(nn.Module):
    def __init__(self, vocab_size = len(TEXT.vocab), pad_idx = TEXT.vocab.stoi[TEXT.pad_token], embedding_dim=300, text_len=50, output_dim=9, feature_size=100):
        super().__init__() #调用nn.Module的构造函数进行初始化
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx) #使用embedding table构建语句到向量的映射
        self.embedding.weight.data.copy_(weight_matrix) #载入由预训练词向量生成的权重矩阵
        self.relu=nn.ReLU() #ReLU函数
        self.bn=nn.BatchNorm1d(num_features=feature_size)
        self.conv1=nn.Conv1d(in_channels=embedding_dim, out_channels=feature_size, kernel_size=2)
        self.conv2=nn.Conv1d(in_channels=embedding_dim, out_channels=feature_size, kernel_size=2)
        self.conv3=nn.Conv1d(in_channels=embedding_dim, out_channels=feature_size, kernel_size=3)
        self.conv4=nn.Conv1d(in_channels=embedding_dim, out_channels=feature_size, kernel_size=3)
    def forward(self, text): #前向传播
        embedded = self.embedding(text)
        embedded = embedded.permute(1, 2, 0)
        out = self.conv1(embedded)
        out = self.bn(out)
        out = nn.ReLU()
        out = self.conv2(embedded)
        out = self.bn(out)
        out = nn.ReLU()
        out = self.conv3(embedded)
        out = self.bn(out)
        out = nn.ReLU()
        out = self.conv4(embedded)
        out = self.bn(out)
        out = nn.ReLU()
        return out
    
class 