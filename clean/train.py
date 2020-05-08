# -*- coding: utf-8 -*-
import torch  # 用于搭建及训练模型
import time  # 用于训练计时
from tqdm import tqdm  # 用于绘制进度条
import torch.nn as nn  # 用于搭建模型
import torch.optim as optim  # 用于生成优化函数
from matplotlib import pyplot as plt #用于绘制误差函数
from model import Model5, Model7, ModelForClustering #导入网络类
from util import TEXT, getTrainIter, getValidIter, calSame, clip_gradient, cluster #导入用到的函数

torch.manual_seed(19260817)  # 设定随机数种子
torch.backends.cudnn.benchmark = True

batch_size = 128
text_len = 7 #单句诗句长度
feature_size = 512 #网络中隐藏层维数
embedding_dim = 256 #词向量维数
#目标分类的数量
#class_size = 84
lr = 0.003 #学习率
decay_factor = 1.004 #学习率梯度衰减参数
betas = (0.9, 0.999) #Adam参数
train_iter = getTrainIter(text_len, batch_size) #获取训练集的迭代器
weight_matrix = TEXT.vocab.vectors  # 构建权重矩阵
loss_function = nn.functional.cross_entropy #使用交叉熵损失函数
vocab_size = len(TEXT.vocab) #词典大小
#模型
if text_len == 5:
    model = Model5(vocab_size=vocab_size, weight_matrix=weight_matrix, pad_idx=TEXT.vocab.stoi[TEXT.pad_token], embedding_dim=embedding_dim, feature_size=feature_size, text_len=text_len)
elif text_len == 7:
    model = Model7(vocab_size=vocab_size, weight_matrix=weight_matrix, pad_idx=TEXT.vocab.stoi[TEXT.pad_token], embedding_dim=embedding_dim, feature_size=feature_size, text_len=text_len)
model.cuda() #使用gpu训练
#将目标分类作为输出结果时的模型, 未能成功训练
#model = ModelForClustering(vocab_size=vocab_size, class_size=class_size, weight_matrix=weight_matrix, pad_idx=TEXT.vocab.stoi[TEXT.pad_token], embedding_dim=embedding_dim, feature_size=feature_size, text_len=text_len)
for params in model.embedding.parameters():
    params.requires_grad = False #固定词嵌入
parameters = filter(lambda p: p.requires_grad, model.parameters())

optimizer = optim.Adam(params=parameters, lr=lr, betas=betas, weight_decay=1e-7)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 / decay_factor)
criterion = nn.CrossEntropyLoss() #交叉熵损失函数

#用于训练过程中检验模型合理性
valist1 = ['白', '日', '不', '到', '处', '青', '春', '恰', '自', '来', '苔', '花', '如', '米', '小', '亦', '学', '牡', '丹', '开']
valist2 = ['鹤', '湖', '东', '去', '水', '茫', '茫', '一', '面', '风', '泾', '接', '魏', '塘', '看', '取', '松', '江', '布', '帆', '至', '鲈', '鱼', '切', '玉', '劝', '郎', '尝']
vac1 = []
vac2 = []
for w in valist1:
    vac1.append(TEXT.vocab.stoi[w])
for w in valist2:
    vac2.append(TEXT.vocab.stoi[w])
vac1 = torch.tensor(vac1, dtype=torch.long).unsqueeze(1)
vac2 = torch.tensor(vac2, dtype=torch.long).unsqueeze(1)
#目标的聚类: 未能成功使用
#dic = cluster(class_size, "dataset/qtrain7")

#训练函数
def fit(epoch):
    start = time.time() #记录训练开始时间
    losses = []
    for i in tqdm(range(1, epoch+1)):
        if i > 1:
            scheduler.step() #learning_rate衰减
        for idx, batch in enumerate(train_iter):
            model.train() #训练模式, 使用dropout
            torch.cuda.empty_cache() #清除显存冗余
            loss = 0
            for j in range(2,5): #生成2-4句
                for k in range(1,text_len+1):
                    if k == 1:
                        state = model.init_hidden(batch.text.size()[1]) #首字的隐藏层用0初始化
                    out, state = model(batch.text.cuda(), state, j, k) #forward
                    loss += loss_function(out, batch.text[text_len*(j-1)+k-1].cuda()) #计算loss
            model.zero_grad()  # 将上次计算得到的梯度值清零
            loss.backward(retain_graph=True)  # 反向传播
            optimizer.step()  # 修正模型
            if idx%10==0: 
                print(loss.item()) #打印损失
                model.eval() #评估模式, 不使用dropout
                for j in range(2,5): #生成2-4句
                    for k in range(1,text_len+1):
                        if k == 1:
                            state = model.init_hidden(1) #首字的隐藏层用0初始化
                        if text_len == 5:
                            out, state = model(vac1.cuda(), state, j, k)
                        if text_len == 7:
                            out, state = model(vac2.cuda(), state, j, k)
                        print(TEXT.vocab.itos[torch.argmax(out)], end=' ')
                    print('\n')
                losses.append(loss.item())
        end = time.time() #记录训练结束时间
        print('Epoch %d' %(i))
        print('Time used: %ds' %(end-start)) #打印训练所用时间
    plt.plot(losses) #绘制训练过程中loss与训练次数的图像'''

#加载checkpoint
model.load_state_dict(torch.load('models/model7.pth'))
for i in range(60):
    fit(5)
    torch.save(model.state_dict(), 'models/modelc'+str(i)+'.pth') #保存checkpoint