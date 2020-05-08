#coding: utf-8
import torch  # 用于搭建及训练模型
import torch.nn as nn  # 用于搭建模型
from torch.autograd import Variable # 用于初始化隐藏状态
from util import TEXT, idx_to_onehot

# 输入: 整首诗(28*batch_size)输出: v1-v_ith_sentence-1(1*embedding dim)
class CSM(nn.Module):
    def __init__(self, embedding_dim, text_len, feature_size):
        super().__init__()  # 调用nn.Module的构造函数进行初始化
        # 使用embedding table构建语句到向量的映射
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
        self.text_len = text_len

    def forward(self, text, ith_sentence):  # 前向传播
        ans = []
        for j in range(1,ith_sentence): #生成v1-v_ith_sentence-1
            out = self.conv1(text[:,:,(j-1)*self.text_len:j*self.text_len])
            out = self.bn(out)
            out = self.relu(out)  # batch_size*feature_size*6
            out = self.conv2(out)
            out = self.bn(out)
            out = self.relu(out)  # batch_size*feature_size*5
            out = self.conv3(out)
            out = self.bn(out)
            out = self.relu(out)  # batch_size*feature_size*3
            if self.text_len == 7:
                out = self.conv4(out)
                out = self.relu(out)  # batch_size*feature_size*1
            ans.append(out.squeeze(2))
        return ans  # batch_size*feature_size*3 

# 输入: vec_i 输出: u_i^j
class RCMUnit(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.relu = nn.LeakyReLU()  # ReLU函数
        self.U = nn.Linear(in_features=feature_size, out_features=feature_size)

    def forward(self, vec):  # 前向传播
        out = self.U(torch.transpose(vec, 0, 1))
        out = self.relu(out)
        return out

# 输入: vec_1-vec_i, 输出: u_i^1-u_i^m组成的list
class RCM(nn.Module):
    def __init__(self, feature_size, num_of_unit):
        super().__init__()
        self.relu = nn.ReLU()  # ReLU函数
        self.M = nn.Linear(in_features=2*feature_size,
                           out_features=feature_size)
        self.U = []
        self.num_of_unit = num_of_unit
        self.feature_size = feature_size
        for i in range(0, num_of_unit):
            self.U.append(RCMUnit(feature_size).cuda())

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
    def __init__(self, vocab_size, feature_size=200, text_len=7):
        super(RGM, self).__init__()
        self.vocab_size = vocab_size
        self.R = nn.Linear(feature_size, feature_size)
        self.H = nn.Linear(feature_size, feature_size)
        self.X = nn.Linear(vocab_size, feature_size)
        self.Y = nn.Linear(feature_size, vocab_size)
        self.r = torch.zeros((feature_size, 1))
        self.relu = nn.ReLU()

    def forward(self, u, w, r, length):  # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        e = idx_to_onehot(w, self.vocab_size, length).cuda()  # X是个list
        ro = self.R(torch.transpose(r, 0, 1)) #feature_size*1
        xo = self.X(torch.transpose(e, 0, 1)) #feature_size*1
        ho = self.H(u) #feature_size*1
        self.r = self.relu(ro+xo+ho)
        y = self.Y(self.r)
        self.r = torch.transpose(self.r, 0, 1)
        return y, self.r #r为更新后状态 batch_size*feature_size*1, y为生成值 1*vocab_size*batch_size

#输入: 28*batch_size(batch_size首诗), ith_sentence(要预测诗句的序号(2-4)), 输出: ans(所预测的诗句的词向量组成的tensor, batch_size*1050(1050为7*embedding_dim)) (需接一个decoder), lst(所预测的诗句(汉字)
class ModelOld(nn.Module):
    def __init__(self, vocab_size, weight_matrix, pad_idx, loss_function, embedding_dim=150, feature_size=200, text_len=7):
        super(ModelOld, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding.weight.data.copy_(weight_matrix)  # 载入由预训练词向量生成的权重矩阵
        #创建用到的三个网络
        self.csm = CSM(embedding_dim=embedding_dim, text_len=text_len, feature_size=feature_size)
        self.rcm = RCM(feature_size=feature_size, num_of_unit=text_len)
        self.rgm = RGM(vocab_size=vocab_size)
        self.loss_function = loss_function
        self.text_len = text_len

    def forward(self, text, ith_sentence, idx): #text 28*batch_size
        embedded = self.embedding(text)
        embedded = embedded.permute(1, 2, 0)
        vecs = self.csm(embedded, ith_sentence) #用CSM生成v_1-v_i vecs为一个list (ith_sentence-1)*
        u = self.rcm(vecs, ith_sentence)
        t = torch.zeros((200, 1), requires_grad=True).cuda()
        loss = torch.zeros([], requires_grad=True).cuda()
        length = text.size()[1]
        w = torch.zeros(length, dtype=torch.long).cuda()
        correct = 0
        for j in range(self.text_len):
            y, t = self.rgm(u[j].cuda(), w.unsqueeze(1), t, length) #y batch_size*vocab_size t feature_size*batch_size
            w = text[(ith_sentence-1)*self.text_len+j] #batch_size
            loss += self.loss_function(y, w)
            correct += int(torch.argmax(y,dim=1).eq(w).sum())
            if idx%100==0:
                print(TEXT.vocab.itos[torch.argmax(y[0],dim=0)], end='')
        if idx%100==0:
            print(correct/length/self.text_len)
        return loss

#整合好的模型
class Model(nn.Module):
    def __init__(self, vocab_size, weight_matrix, pad_idx, embedding_dim=150, feature_size=200, text_len=7, dropout=0.2):
        super(Model, self).__init__()
        self.feature_size = feature_size
        self.vocab_size = vocab_size
        self.text_len = text_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding.weight.data.copy_(weight_matrix)  #载入由预训练词向量生成的权重矩阵
        self.relu = nn.LeakyReLU()  #ReLU函数
        self.bn = nn.BatchNorm1d(num_features=feature_size)
        self.d = nn.Dropout(p=dropout)
        #CSM所用的卷积层
        self.conv1 = nn.Conv1d(in_channels=embedding_dim,
                               out_channels=feature_size, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=feature_size,
                               out_channels=feature_size, kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=feature_size,
                               out_channels=feature_size, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=feature_size,
                               out_channels=feature_size, kernel_size=3)
        self.M = nn.Linear(in_features=2*feature_size,
                           out_features=feature_size)
        #RCM
        self.U1 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U2 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U3 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U4 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U5 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U6 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U7 = nn.Linear(in_features=feature_size, out_features=feature_size)
        #RGM
        self.R = nn.Linear(feature_size, feature_size)
        self.H = nn.Linear(feature_size, feature_size)
        self.X = nn.Linear(embedding_dim, feature_size)
        self.Y = nn.Linear(feature_size, vocab_size)

    def forward(self, text, state, ith_sentence, ith_character): #text 28*batch_size
        #---------------------------------------------------------------------#
        #词嵌入
        embedded = self.embedding(text)
        embedded = embedded.permute(1, 2, 0)
        #---------------------------------------------------------------------#
        #CSM部分
        vecs = []
        for j in range(1,ith_sentence): #生成v1-v_ith_sentence-1
            out = self.conv1(embedded[:,:,(j-1)*self.text_len:j*self.text_len])
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
        #---------------------------------------------------------------------#
        #RCM部分
        h = torch.zeros((vecs[0].size()[0], self.feature_size)).cuda()
        for i in range(0, ith_sentence-1):
            out = torch.cat((vecs[i], h), dim=1)
            out = self.M(out)
            h = self.relu(out)
        if ith_character == 1:
            rcmout = self.U1(h).cuda()
        elif ith_character == 2:
            rcmout = self.U2(h).cuda()
        elif ith_character == 3:
            rcmout = self.U3(h).cuda()
        elif ith_character == 4:
            rcmout = self.U4(h).cuda()
        elif ith_character == 5:
            rcmout = self.U5(h).cuda()
        elif ith_character == 6:
            rcmout = self.U6(h).cuda()
        elif ith_character == 7:
            rcmout = self.U7(h).cuda()
        rcmout = self.relu(rcmout)
        #---------------------------------------------------------------------#
        #RGM部分
        w = text[self.text_len*(ith_sentence-1)+ith_character-2]
        middle = self.R(state)
        w = self.embedding(w) #单字
        state = self.relu(middle+self.X(w)+self.H(rcmout))
        y = self.Y(state)
        return y, state
    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.feature_size)).cuda()

#将结果分类后使用的模型
class ModelForClustering(nn.Module):
    def __init__(self, vocab_size, class_size, weight_matrix, pad_idx, embedding_dim=150, feature_size=200, text_len=7, dropout=0.2):
        super(ModelForClustering, self).__init__()
        self.feature_size = feature_size
        self.vocab_size = vocab_size
        self.class_size = class_size
        self.text_len = text_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding.weight.data.copy_(weight_matrix)  # 载入由预训练词向量生成的权重矩阵
        self.relu = nn.LeakyReLU()  # ReLU函数
        self.softmax = nn.Softmax(dim=1)  # ReLU函数
        self.d = nn.Dropout(p=dropout)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim,
                               out_channels=feature_size, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=feature_size,
                               out_channels=feature_size, kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=feature_size,
                               out_channels=feature_size, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=feature_size,
                               out_channels=feature_size, kernel_size=3)
        self.M = nn.Linear(in_features=2*feature_size,
                           out_features=feature_size)
        self.U1 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U2 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U3 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U4 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U5 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U6 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.U7 = nn.Linear(in_features=feature_size, out_features=feature_size)
        self.R = nn.Linear(feature_size, feature_size)
        self.H = nn.Linear(feature_size, feature_size)
        self.X = nn.Linear(embedding_dim, feature_size)
        self.Y = nn.Linear(feature_size, class_size)

    def forward(self, text, state, ith_sentence, ith_character): #text 28*batch_size
        #---------------------------------------------------------------------#
        #词嵌入
        embedded = self.embedding(text)
        embedded = embedded.permute(1, 2, 0)
        #---------------------------------------------------------------------#
        #CSM部分
        vecs = []
        for j in range(1,ith_sentence): #生成v1-v_ith_sentence-1
            out = self.conv1(embedded[:,:,(j-1)*self.text_len:j*self.text_len])
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
        #---------------------------------------------------------------------#
        #RCM部分
        h = torch.zeros((vecs[0].size()[0], self.feature_size))
        for i in range(0, ith_sentence-1):
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
        #---------------------------------------------------------------------#
        #RGM部分
        length = text.size()[1]
        w = self.embedding(text[self.text_len*(ith_sentence-1)+ith_character-2])
        middle = self.R(state)
        state = self.relu(middle+self.X(w)+self.H(rcmout))
        y = self.Y(state)
        y = self.softmax(y)
        return y, state
    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.feature_size)).cuda()