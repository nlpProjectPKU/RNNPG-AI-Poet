# -*- coding: utf-8 -*-
import torch  # 用于搭建及训练模型
import time  # 用于训练计时
from tqdm import tqdm  # 用于绘制进度条
import torch.nn as nn  # 用于搭建模型
import torch.optim as optim  # 用于生成优化函数
from matplotlib import pyplot as plt #用于绘制误差函数
from model import Model, ModelForClustering
from util import TEXT, getTrainIter, getValidIter, calSame, clip_gradient, cluster
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.manual_seed(19260817)  # 设定随机数种子
torch.backends.cudnn.benchmark = True  # 保证可复现性

batch_size = 1
text_len = 7
feature_size = 200
embedding_dim = 150
class_size = 84
valid_iter = getValidIter(text_len, batch_size)
train_iter = getTrainIter(text_len, batch_size)
weight_matrix = TEXT.vocab.vectors  # 构建权重矩阵
loss_function = nn.functional.cross_entropy #使用交叉熵损失函数
vocab_size = len(TEXT.vocab)
model = ModelForClustering(vocab_size=vocab_size, class_size=class_size, weight_matrix=weight_matrix, pad_idx=TEXT.vocab.stoi[TEXT.pad_token], embedding_dim=embedding_dim, feature_size=feature_size, text_len=text_len)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
model

#用于检验模型合理性
valist1 = ['白', '日', '不', '到', '处', '青', '春', '恰', '自', '来', '苔', '花', '如', '米', '小', '亦', '学', '牡', '丹', '开']
valist2 = ['人', '间', '七', '月', '炎', '云', '升', '碧', '树', '黄', '鹂', '亦', '可', '人', '哑', '咤', '一', '声', '行', '道', '外', '不', '知','身', '在', '故', '园', '春']
vac1 = []
vac2 = []
for w in valist1:
    vac1.append(TEXT.vocab.stoi[w])
for w in valist2:
    vac2.append(TEXT.vocab.stoi[w])
vac1 = torch.tensor(vac1, dtype=torch.long).unsqueeze(1)
vac2 = torch.tensor(vac2, dtype=torch.long).unsqueeze(1)
dic = cluster(class_size, "dataset/qtrain7")

def fit(epoch):
    model.train()
    start = time.time() #记录训练开始时间
    losses = []
    for i in tqdm(range(1, epoch+1)):
        for idx, batch in enumerate(train_iter):
            torch.cuda.empty_cache()
            state = torch.zeros((batch.text.size()[1], feature_size), requires_grad=True)
            loss = 0
            for j in range(2,5): #生成2-4句
                for k in range(1,text_len+1):
                    #if k == 1:
                    #    state = torch.zeros((1, feature_size), requires_grad=True).cuda()
                    out, state = model(batch.text, state, j, k)
                    #loss += loss_function(out, batch.text[text_len*(j-1)+k-1].cuda())
                    loss += -torch.log(out[:,dic[TEXT.vocab.itos[int(batch.text[text_len*(j-1)+k-1])]][0]]*dic[TEXT.vocab.itos[int(batch.text[text_len*(j-1)+k-1])]][1])
            model.zero_grad()  # 将上次计算得到的梯度值清零
            loss.backward()  # 反向传播
            #clip_gradient(optimizer, 0.1)
            optimizer.step()  # 修正模型
            if idx%100==0:
                print(loss.item()) #打印损失
                '''
                state = torch.zeros((1, feature_size), requires_grad=True).cuda()
                for j in range(2,5): #生成2-4句
                    for k in range(1,text_len+1):
                        if text_len == 5:
                            out, state = model(vac1.cuda(), state, j, k)
                        if text_len == 7:
                            out, state = model(vac2.cuda(), state, j, k)
                        print(TEXT.vocab.itos[torch.argmax(out)], end=' ')
                    print('\n')
                '''
                losses.append(loss.item())
                '''
                print("validation")
                correct = 0 #预测正确的字数
                total = 0 #需预测的字数的总数
                for idx2, batch2 in enumerate(valid_iter):
                    for j2 in range(1,4):
                        for k2 in range(1,text_len+1):
                            if k2 == 1:
                                state2 = torch.zeros((1, feature_size), requires_grad=True).cuda()
                            out2, state2 = model(batch2.text.cuda(), state2, j2+1, k2)
                            correct += calSame(out2, batch2.text[k].cuda())
                    total += batch2.text.size()[1]*text_len*3
                print(correct/total)
                '''
        end = time.time() #记录训练结束时间
        print('Epoch %d' %(i))
        print('Time used: %ds' %(end-start)) #打印训练所用时间
    plt.plot(losses) #绘制训练过程中loss与训练次数的图像'''

torch.cuda.empty_cache()
#model.load_state_dict(torch.load('models/model.pth'))
for i in range(60):
    fit(1)
    torch.save(model.state_dict(), 'models/modelc'+str(i)+'.pth')