# -*- coding: utf-8 -*-
import torch  # 用于搭建及训练模型
import time  # 用于训练计时
from torchtext import data  # 用于生成数据集
from torchtext.vocab import Vectors  # 用于载入预训练词向量
from tqdm import tqdm  # 用于绘制进度条
from torchtext.data import BucketIterator  # 用于生成训练和测试所用的迭代器
import torch.nn as nn  # 用于搭建模型
import torch.optim as optim  # 用于生成优化函数
import os.path as path
import codecs
from matplotlib import pyplot as plt #用于绘制误差函数

torch.manual_seed(19260817)  # 设定随机数种子
torch.backends.cudnn.deterministic = True  # 保证可复现性
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tokenize(x): return x.split()

wordvecPath = "word2vec.vector"
dataPath = "dataset/"
TEXT = data.Field(sequential=True, tokenize=tokenize)

class Dataset(data.Dataset):
    name = 'Dataset'
    def __init__(self, fin, text_field):
        fields = [("text", text_field)]
        examples = []
        print('read data from {}'.format(path))
        for line in fin:
            examples.append(data.Example.fromlist([line], fields))
        super(Dataset, self).__init__(examples, fields) #生成标准dataset


def getDataIter(fin, fiveOrSeven):
    data = Dataset(fin, TEXT)
    vectors = Vectors(name='word2vec.vector')
    TEXT.build_vocab(data, vectors=vectors, unk_init = torch.Tensor.normal_) #构建映射,设定最低词频为5
    return BucketIterator(dataset=data, batch_size=batch_size, shuffle=True)

def getTrainIter(fiveOrSeven):
    assert fiveOrSeven == 5 or fiveOrSeven == 7
    trainfin = codecs.open(path.join(dataPath, "qtrain"+str(fiveOrSeven)), 'r', encoding = 'utf-8')
    return getDataIter(trainfin, fiveOrSeven)

def getTestIter(fiveOrSeven):
    assert fiveOrSeven == 5 or fiveOrSeven == 7
    testfin = codecs.open(path.join(dataPath, "qtest"+str(fiveOrSeven)), 'r', encoding = 'utf-8')
    return getDataIter(testfin, fiveOrSeven)

def getValidIter(fiveOrSeven):
    assert fiveOrSeven == 5 or fiveOrSeven == 7
    validfin = codecs.open(path.join(dataPath, "qvalid"+str(fiveOrSeven)), 'r', encoding = 'utf-8')
    return getDataIter(validfin, fiveOrSeven)

batch_size = 16
train_iter = getTrainIter(7)
weight_matrix = TEXT.vocab.vectors  # 构建权重矩阵
weight_matrix.cuda()

"""一堆生成one-hot表示的函数"""

def idx_to_onehot(i):
    res = torch.zeros((len(TEXT.vocab), 1))
    res[i] = 1
    return res

def char_to_onehot(c):
    res = torch.zeros((len(TEXT.vocab), 1))
    res[TEXT.vocab.stoi[c]] = 1
    return res

def sentence_to_onehot(idx):
    res = torch.zeros((6*len(TEXT.vocab), batch_size), dtype=torch.long)
    for i in range(batch_size):
        for j in range(6):
            res[idx[j][i]+j*len(TEXT.vocab)][i] = 1
    return res

"""全部的网络"""

# 输入: 整首诗(28*batch_size)输出: v1-v_ith_sentence-1(1*embedding dim)
class CSM(nn.Module):
    def __init__(self, vocab_size=len(TEXT.vocab), pad_idx=TEXT.vocab.stoi[TEXT.pad_token], embedding_dim=150, text_len=7, feature_size=200):
        super().__init__()  # 调用nn.Module的构造函数进行初始化
        # 使用embedding table构建语句到向量的映射
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding.weight.data.copy_(weight_matrix)  # 载入由预训练词向量生成的权重矩阵
        self.embedding.to(device)
        self.relu = nn.LeakyReLU()  # ReLU函数
        self.bn = nn.BatchNorm1d(num_features=feature_size)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim,
                               out_channels=feature_size, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=feature_size,
                               out_channels=feature_size, kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=feature_size,
                               out_channels=feature_size, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=feature_size,
                               out_channels=feature_size, kernel_size=3)

    def forward(self, text, ith_sentence):  # 前向传播
        ans = []
        if training:
            for j in range(1,ith_sentence): #生成v1-v_ith_sentence-1
                embedded = self.embedding(text[(j-1)*7:j*7])
                # print(text)
                # batch_size*embedding_dim*text_len(7)
                embedded = embedded.permute(1, 2, 0)
                out = self.conv1(embedded)
                out = self.bn(out)
                out = self.relu(out)  # batch_size*feature_size*6
                out = self.conv2(out)
                out = self.bn(out)
                out = self.relu(out)  # batch_size*feature_size*5
                out = self.conv3(out)
                out = self.bn(out)
                out = self.relu(out)  # batch_size*feature_size*3
                out = self.conv4(out)
                out = self.relu(out)  # batch_size*feature_size*1
                ans.append(out.squeeze())
        return ans  # batch_size*feature_size*3 

# 输入: vec_i 输出: u_i^j
class RCMUnit(nn.Module):
    def __init__(self, feature_size=200):
        super().__init__()
        self.relu = nn.ReLU()  # ReLU函数
        self.U = nn.Linear(in_features=feature_size, out_features=feature_size)

    def forward(self, vec):  # 前向传播
        out = self.U(torch.transpose(vec, 0, 1))
        out = self.relu(out)
        return out

# 输入: vec_1-vec_i, 输出: u_i^1-u_i^m组成的list
class RCM(nn.Module):
    def __init__(self, feature_size=200, num_of_unit=7):
        super().__init__()
        self.relu = nn.ReLU()  # ReLU函数
        self.M = nn.Linear(in_features=2*feature_size,
                           out_features=feature_size)
        self.U = []
        self.num_of_unit = num_of_unit
        self.feature_size = feature_size
        for i in range(0, num_of_unit):
            self.U.append(RCMUnit().cuda())

    def forward(self, vecs, ith_sentence):  # 前向传播
        ans = []
        h = torch.zeros((vecs[0].size()[0], self.feature_size)).cuda()
        for i in range(0, ith_sentence-1):
            out = torch.cat((vecs[i], h), dim=1)
            out = self.M(out)
            h = self.relu(out)
        for j in range(0, self.num_of_unit):
            out = self.U[j](torch.transpose(h, 0, 1))
            ans.append(out)
        return ans

# 输入: u_i^j,w_j 输出: 最可能的第j+1个字
class RGM(nn.Module):
    def __init__(self, vocab_size=len(TEXT.vocab), feature_size=200, text_len=7):
        super(RGM, self).__init__()
        self.vocab_size = vocab_size
        self.R = nn.Linear(feature_size, feature_size)
        self.H = nn.Linear(feature_size, feature_size)
        self.X = nn.Linear(vocab_size, feature_size)
        self.Y = nn.Linear(feature_size, vocab_size)
        self.r = torch.zeros((feature_size, 1))
        self.relu = nn.ReLU()

    def forward(self, u, w, r):  # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        e = idx_to_onehot(w).cuda()  # X是个list
        ro = self.R(torch.transpose(r, 0, 1)) #feature_size*1
        xo = self.X(torch.transpose(e, 0, 1)) #feature_size*1
        ho = self.H(u) #feature_size*1
        self.r = self.relu(ro+xo+ho)
        y = self.Y(self.r)
        self.r = torch.transpose(self.r, 0, 1)
        return y, self.r #r为更新后状态 batch_size*feature_size*1, y为生成值 1*vocab_size*batch_size



#输入: 28*batch_size(batch_size首诗), ith_sentence(要预测诗句的序号(2-4)), 输出: ans(所预测的诗句的词向量组成的tensor, batch_size*1050(1050为7*embedding_dim)) (需接一个decoder), lst(所预测的诗句(汉字)
class Model(nn.Module):
    def __init__(self, vocab_size=len(TEXT.vocab), feature_size=200, text_len=7):
        super(Model, self).__init__()
        #创建用到的三个网络
        self.csm = CSM()
        self.rcm = RCM()
        self.rgm = RGM()

    def forward(self, text, ith_sentence): #text 28*batch_size
        vecs = self.csm(text, ith_sentence) #用CSM生成v_1-v_i vecs为一个list (ith_sentence-1)*
        u = self.rcm(vecs, ith_sentence)  #
        t = torch.zeros((200, 1), requires_grad=True).cuda()
        #使用以下被注释的代码可以实现向量化生成整个batch的预测值
        loss = torch.zeros([], requires_grad=True).cuda()
        w = 0
        for j in range(7):
            #print(u[j].size())
            y, t = self.rgm(u[j].cuda(), w, t) #y batch_size*vocab_size t feature_size*batch_size
            w = text[(ith_sentence-1)*7+j] #batch_size
            loss += loss_function(y, w)
            #print(j)
            print(TEXT.vocab.itos[torch.argmax(y[0],dim=0)], end='')
        print('\n')
        return loss

#用于测试 生成输入诗句的下一句
def put(str, ith_sentence):
    s = tokenize(str)
    l = []
    print(s)
    for w in s:
        l.append([TEXT.vocab.stoi[w]])
    lt = torch.tensor(l)
    return model(lt.cuda(), ith_sentence)


def fit(epoch):
    start = time.time() #记录训练开始时间
    losses = []
    for i in tqdm(range(1, epoch+1)):
        for idx, batch in enumerate(train_iter):
            for j in range(1,4): #生成2-4句
                model.zero_grad()  # 将上次计算得到的梯度值清零
                model.train()  # 将模型设为训练模式'
                loss = model(batch.text.cuda(), j+1)
                loss.backward()  # 反向传播
                optimizer.step()  # 修正模型
            if idx%10==0:
                print(loss.item()) #打印损失
            losses.append(loss.item())
        end = time.time() #记录训练结束时间
        print('Epoch %d' %(i))
        print('Time used: %ds' %(end-start)) #打印训练所用时间
    plt.plot(losses) #绘制训练过程中loss与训练次数的图像'''



model = Model()
loss_function = nn.functional.cross_entropy #使用交叉熵损失函数
optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
#loss_function = nn.functional.nll_loss
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)  # 使用Adam作为优化器

model.cuda()

model.load_state_dict(torch.load('models/modeln9.pth'))
training = True
fit(1)
#保存模型
#torch.save(model.state_dict(), 'models/model.pth')