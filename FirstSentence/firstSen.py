# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as DataSet
from torch.autograd import Variable

import os
import random
import numpy as np

path = {
    # Ping/Ze tonal of all characters
    "TONAL_PATH": "./dataset/pingshui.txt",
    # category of words
    "SHIXUEHANYING_PATH": "./dataset/shixuehanying.txt",
    # first sentences
    "FIRSTSEN_PATH": ["./poem_lm/qtais_tab.txt"],
}

fixLen = 5  #length of poem
epoch_time = 10
teacher_forcing = 0.5
useCUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Encoder_args = {
    "nLayers": 1,
    "hiddenSize": 32,
    "dictionSize": None,
    "batchSize": 16,
    "learningRate": 0.0001,
}

Decoder_args = {
    "nLayers": 1,
    "hiddenSize": Encoder_args["hiddenSize"],
    "outputSize": None,
    "batchSize": Encoder_args["batchSize"],
    "dropout": 0.1,
    "learningRate": 0.0001,
}

def getTone():
    # get tonal dictionary of each character
    ping = []
    ze = []
    with open(path["TONAL_PATH"], 'r') as f:
        isPing = False
        for line in  f.readlines():
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
    return {"Ping":ping, "Ze":ze}

def getShixuehanying():
    # get words list of shixuehanying
    classes = []
    labels = []
    words = []
    with open(path["SHIXUEHANYING_PATH"], 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                if line[0] == '<':
                    if line[1] == 'b':  # <begin>
                        titles = line.split('\t')
                        classes.append(titles[2])
                else:
                    line = line.split('\t')
                    if len(line) == 3:
                        tmp = line[2].split(' ')
                        tmp.append(line[1])
                        labels.append(line[1])
                        words.append(tmp)
    return labels, words

def getSen():
    # get first sentense of poem
    data = []
    for dir in path["FIRSTSEN_PATH"]:
        with open(dir, 'r', encoding="UTF-8-sig") as f:
            for line in f.readlines():
                tmp = line.split()
                if len(tmp) == fixLen:
                    data.append(tmp)
    return data

def getDiction(data_set):
    # bulid diction of data
    char2id_dic = {}
    id2char_dic = {}
    for sen in data_set:
        for w in sen:
            if w not in char2id_dic.keys():
                _id = len(char2id_dic)
                char2id_dic[w] = _id
                id2char_dic[_id] = w
    Encoder_args["dictionSize"] = Decoder_args["outputSize"] = len(char2id_dic)
    return char2id_dic, id2char_dic

def char2id(sen, char2id_dic):
    l = []
    for i in sen:
        l.append(char2id_dic[i])
    return np.array(l)

def id2char(sen, id2char_dic):
    l = []
    for i in sen:
        l.append(id2char_dic[i])
    return l

def PrepareData():
    data = getSen()
    char2id_dic, id2char_dic = getDiction(data)
    data = [char2id(i, char2id_dic) for i in data]
    
    permutation = np.random.permutation(range(len(data)))
    data = [data[i] for i in permutation]
    test_size = len(data) // 10
    train_data = np.array(data[test_size: ])
    valid_data = np.array(data[: test_size])
    train_dataset = DataSet.TensorDataset(torch.LongTensor(train_data))
    valid_dataset = DataSet.TensorDataset(torch.LongTensor(valid_data))
    train_loader = DataSet.DataLoader(train_dataset, batch_size=Encoder_args["batchSize"], shuffle = True)
    valid_loader = DataSet.DataLoader(valid_dataset, batch_size=Encoder_args["batchSize"], shuffle = True)
    return train_loader, valid_loader, char2id_dic, id2char_dic

train_loader, valid_loader, char2id_dic, id2char_dic = PrepareData()


class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        self.n_layers = Encoder_args["nLayers"]
        self.hidden_size = Encoder_args["hiddenSize"]
        self.input_size = Encoder_args["dictionSize"]
        self.batch_size = Encoder_args["batchSize"]
        
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, batch_first=True)
    
    def forward(self, x, hidden):
        # batch, seq_len
        embed = self.embedding(x)
        # batch, seq_len, embed_dim
        output = embed
        output, hidden = self.gru(output, hidden)
        # output: batch, seq_len, hidden_size
        # hidden: layer, batch, hidden_size
        return output, hidden
    
    def initHidden(self, batch=Encoder_args["batchSize"]):
        return Variable(torch.zeros(self.n_layers, batch, self.hidden_size))

class AttnDecoderRNN(nn.Module):
    def __init__(self):
        super(AttnDecoderRNN, self).__init__()
        self.n_layers = Decoder_args["nLayers"]
        self.hidden_size = Decoder_args["hiddenSize"]
        self.output_size = Decoder_args["outputSize"]
        self.dropout_p = Decoder_args["dropout"]
        self.batch_size = Decoder_args["batchSize"]
        self.max_len = fixLen
        
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * (self.n_layers + 1), self.max_len)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x, hidden, encoder_output):
        # batch, seq_len
        embed = self.embedding(x)
        # batch, seq_len, hidden
        embed = embed[:, 0, :]
        # batch, hidden
        embed = self.dropout(embed)
        # hidden: n_layer, batch, hidden
        tmp_for_tran = torch.transpose(hidden, 0, 1).contiguous()
        tmp_for_tran = tmp_for_tran.view(tmp_for_tran.size()[0], -1)
        hidden_attn = tmp_for_tran
        # attention
        input_to_attn = torch.cat((embed, hidden_attn), 1)
        # batch, hidden * (1 + n_layer)
        attn_weights = F.softmax(self.attn(input_to_attn), dim=1)
        # batch, fixLen
        attn_weights = attn_weights[:,:encoder_output.size()[1]]
        attn_weights = attn_weights.unsqueeze(1)
        # batch, 1, fixLen
        attn_applied = torch.bmm(attn_weights, encoder_output)
        # batch, 1, hidden
        output = torch.cat((embed, attn_applied[:, 0, :]), 1)
        # batch, hidden * 2
        output = self.attn_combine(output).unsqueeze(1)
        output = F.relu(output)
        output = self.dropout(output)
        output, hidden = self.gru(output, hidden)
        # output: batch, seq_len, hidden
        # hidden: n_layers, batch, hidden
        output = F.log_softmax(self.out(output[:, -1, :]), dim=1)
        # batch, output_size
        return output, hidden, attn_weights
        
    def initHidden(self, batch = Decoder_args["batchSize"]):
        return Variable(torch.zeros(self.n_layers, batch, self.hidden_size))


def rightness(prediction, labels):
    pred = torch.max(prediction.data, 1)[1]
    rights = pred.eq(labels.data).sum()
    return rights, len(labels)


encoder = EncoderRNN()
decoder = AttnDecoderRNN()
if useCUDA:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
criterion = nn.NLLLoss()
def train(train_loader, valid_loader):

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=Encoder_args["learningRate"])
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=Decoder_args["learningRate"])

    for epoch in range(epoch_time):
        decoder.train()
        print_loss_total = 0
        for (data, ) in train_loader:
            if data.size()[0] < Encoder_args["batchSize"]:
                continue
            input_var = Variable(data).cuda() if useCUDA else Variable(data)
            # batch, fixLen
            target_var = Variable(data).cuda() if useCUDA else Variable(data)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            loss = 0
            encoder_hidden = encoder.initHidden()
            encoder_outputs, encoder_hidden = encoder(input_var, encoder_hidden)
            # output: batch, fixLen, hidden
            # hidden: n_layer, batch, hidden
            decoder_input = Variable(torch.LongTensor([[0]] * target_var.size()[0]))
            decoder_input = decoder_input.cuda() if useCUDA else decoder_input
            decoder_hidden = encoder_hidden
            
            use_teacher_forcing = True if random.random() < teacher_forcing else False
            if use_teacher_forcing:
                for di in range(fixLen):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    loss += criterion(decoder_output, target_var[:, di])
                    decoder_input = target_var[:, di].unsqueeze(1)  # teaching forcing
            else:
                for di in range(fixLen):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    topv, topi = decoder_output.data.topk(1, dim=1)
                    ni = topi[:, 0]
                    decoder_input = Variable(ni.unsqueeze(1))
                    decoder_input = decoder_input.cuda() if useCUDA else decoder_input
                    loss += criterion(decoder_output, target_var[:, di])
                    
            loss.backward()
            loss = loss.cpu() if useCUDA else loss
            encoder_optimizer.step()
            decoder_optimizer.step()
            print_loss_total += loss.data.numpy()
            
        print_loss_avg = print_loss_total / len(train_loader)
        valid_loss = 0
        rights = []
        decoder.eval()  # close dropout
        for (data, ) in valid_loader:
            if data.size()[0] < Encoder_args["batchSize"]:
                continue
            input_var = Variable(data).cuda() if useCUDA else Variable(data)
            # batch, fixLen
            target_var = Variable(data).cuda() if useCUDA else Variable(data)
            # encode
            loss = 0
            encoder_hidden = encoder.initHidden()
            encoder_outputs, encoder_hidden = encoder(input_var, encoder_hidden)
            # output: batch, fixLen, hidden
            # hidden: n_layer, batch, hidden
            # decode
            decoder_input = Variable(torch.LongTensor([[0]] * target_var.size()[0]))
            decoder_input = decoder_input.cuda() if useCUDA else decoder_input
            decoder_hidden = encoder_hidden
            for di in range(fixLen):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1, dim=1)
                ni = topi[:, 0]
                decoder_input = Variable(ni.unsqueeze(1))
                decoder_input = decoder_input.cuda() if useCUDA else decoder_input
                loss += criterion(decoder_output, target_var[:, di])
                right = rightness(decoder_output, target_var[:, di])
                rights.append(right)
            loss = loss.cpu() if useCUDA else loss
            valid_loss += loss.data.numpy()
        right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
        print_valid_loss = valid_loss / len(valid_loader)
        print("epoch:{} train loss:{} valid loss:{} valid accuary:{}".format(epoch, print_loss_avg, print_valid_loss, right_ratio))
        
train(train_loader, valid_loader)


# generator
# 0 represents either, 1 represents Ping, -1 represents Ze
FIVE_PINGZE = [[0, -1, 1, 1, -1], [1, 1, -1, -1, 1], [0, 1, 1, -1, -1], [0, -1, -1, 1, 1],
               [0, -1, -1, 1, 1], [1, 1, -1, -1, 1], [0, 1, 1, -1, -1], [0, -1, -1, 1, 1],
               [0, 1, 1, -1, -1], [0, -1, -1, 1, 1], [0, -1, 1, 1, -1], [1, 1, -1, -1, 1],
               [1, 1, -1, -1, 1], [0, -1, -1, 1, 1], [0, -1, 1, 1, -1], [1, 1, -1, -1, 1]]

SEVEN_PINGZE = [[0, 1, 0, -1, -1, 1, 1], [0, -1, 1, 1, -1, -1, 1], [0, -1, 0, 1, 1, -1, -1], [0, 1, 0, -1, -1, 1, 1],
                [0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, -1, -1, 1], [0, -1, 0, 1, 1, -1, -1], [0, 1, 0, -1, -1, 1, 1],
                [0, -1, 1, 1, -1, -1, 1], [0, 1, 0, -1, -1, 1, 1], [0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, -1, -1, 1],
                [0, -1, 0, 1, 1, -1, -1], [0, 0, -1, -1, -1, 1, 1], [0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, -1, -1, 1]]

tone = getTone()
def judgeTonalPattern(sen):
    # judge if the given sentence follows the given tonal pattern
    TONAL = FIVE_PINGZE if fixLen == 5 else SEVEN_PINGZE
    flag = False
    for form in TONAL:
        flag1 = True
        for _id,w in enumerate(sen):
            if form[_id] == 0:
                continue
            if form[_id] == 1 and w not in tone["Ping"]:
                flag1 = False
                break
            elif form[_id] == 0 and w not in tone["Ze"]:
                flag1 = False
                break
        if flag1:
            flag = True
            break
    return flag

def score(sen):
    # score a poem
    data = np.array(char2id(sen, char2id_dic))
    data = torch.LongTensor(data).unsqueeze(0)
    input_var = Variable(data).cuda() if useCUDA else Variable(data)
    target_var = Variable(data).cuda() if useCUDA else Variable(data)
    # batch, len
    # encode
    loss = 0
    encoder_outputs = Variable(torch.zeros(1, len(sen), Encoder_args["hiddenSize"]))
    encoder_hidden = Variable(torch.zeros(Encoder_args["nLayers"], 1, Encoder_args["hiddenSize"]))
    # output: batch, fixLen, hidden
    # hidden: n_layer, batch, hidden
    # decode
    decoder_input = Variable(torch.LongTensor([[0]] * target_var.size()[0]))
    decoder_input = decoder_input.cuda() if useCUDA else decoder_input
    decoder_hidden = encoder_hidden
    for di in range(len(sen)):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.data.topk(1, dim=1)
        ni = topi[:, 0]
        decoder_input = Variable(ni.unsqueeze(1))
        decoder_input = decoder_input.cuda() if useCUDA else decoder_input
        loss += criterion(decoder_output, target_var[:, di])
        loss = loss.cpu() if useCUDA else loss
    return loss

def generator(topic, topn = 5):
    # generate first Sentence
    labels, words = getShixuehanying()
    word_collection = []
    for _id,l in enumerate(labels):
        if l in topic:
             word_collection += words[_id]
    
    for w in word_collection:  # erase words not in diction
        flag = False
        for char in w:
            if char not in char2id_dic.keys():
                flag = True
                break
        if flag:
            word_collection.remove(w)

    hypo = []
    queue = []
    queue += word_collection
    while len(queue) > 0:
        expend = queue[0]
        queue.remove(queue[0])
        tmp = []  # possible nodes in expend
        for w in word_collection:
            if w in expend:
                continue
            evaluate = expend + w
            if len(evaluate) > fixLen or not judgeTonalPattern(evaluate):
                continue
            if len(evaluate) == fixLen:
                print(evaluate)
                cost = score(evaluate)
                hypo.append((evaluate, cost))
                hypo.sort(key=lambda x: x[1])
                hypo = hypo[:topn]
                continue
            cost = score(evaluate)
            tmp.append((evaluate, cost))
        tmp.sort(key=lambda x: x[1])
        queue += [tmp[i][0] for i in range(min(topn, len(tmp)))]
    return hypo

# generator(["不寝"])