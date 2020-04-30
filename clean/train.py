# -*- coding: utf-8 -*-
import torch  # 用于搭建及训练模型
import time  # 用于训练计时
from tqdm import tqdm  # 用于绘制进度条
import torch.nn as nn  # 用于搭建模型
import torch.optim as optim  # 用于生成优化函数
from matplotlib import pyplot as plt #用于绘制误差函数
from model import Model, ModelAll
from util import TEXT, getTrainIter, getValidIter

torch.manual_seed(19260817)  # 设定随机数种子
torch.backends.cudnn.deterministic = True  # 保证可复现性

batch_size = 128
text_len = 5
feature_size = 200
embedding_dim = 150
valid_iter = getValidIter(text_len, batch_size)
train_iter = getTrainIter(text_len, batch_size)
weight_matrix = TEXT.vocab.vectors  # 构建权重矩阵

loss_function = nn.functional.cross_entropy #使用交叉熵损失函数
model = ModelAll(vocab_size=len(TEXT.vocab),weight_matrix=weight_matrix, pad_idx=TEXT.vocab.stoi[TEXT.pad_token], loss_function=loss_function, embedding_dim=embedding_dim, feature_size=feature_size, text_len=text_len)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
model.cuda()

def fit(epoch):
    model.train()
    start = time.time() #记录训练开始时间
    losses = []
    for i in tqdm(range(1, epoch+1)):
        for idx, batch in enumerate(train_iter):
            for j in range(1,4): #生成2-4句
                model.zero_grad()  # 将上次计算得到的梯度值清零
                loss, correct = model(batch.text.cuda(), j+1, idx)
                loss.backward()  # 反向传播
                optimizer.step()  # 修正模型
            if idx%100==0:
                print(loss.item()) #打印损失
                losses.append(losses)
                print("validation")
                for idx2, batch2 in enumerate(valid_iter):
                    for j2 in range(1,4):
                        loss, correct = model(batch2.text.cuda(), j2+1, idx2)
        end = time.time() #记录训练结束时间
        print('Epoch %d' %(i))
        print('Time used: %ds' %(end-start)) #打印训练所用时间
    plt.plot(losses) #绘制训练过程中loss与训练次数的图像'''

model.load_state_dict(torch.load('models/model.pth'))
fit(10)
#torch.save(model.state_dict(), 'models/model.pth')