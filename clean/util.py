# -*- coding: utf-8 -*-
import torch  # 用于搭建及训练模型
from torchtext import data  # 用于生成数据集
from torchtext.vocab import Vectors  # 用于载入预训练词向量
from torchtext.data import BucketIterator  # 用于生成训练和测试所用的迭代器
import os.path as path
import codecs

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

def getDataIter(fin, fiveOrSeven, batch_size):
    data = Dataset(fin, TEXT)
    vectors = Vectors(wordvecPath)
    TEXT.build_vocab(data, vectors=vectors, unk_init = torch.Tensor.normal_) #构建映射,设定最低词频为5
    return BucketIterator(dataset=data, batch_size=batch_size, shuffle=True)

def getTrainIter(fiveOrSeven, batch_size):
    assert fiveOrSeven == 5 or fiveOrSeven == 7
    trainfin = codecs.open(path.join(dataPath, "qtrain"+str(fiveOrSeven)), 'r', encoding = 'utf-8')
    return getDataIter(trainfin, fiveOrSeven, batch_size)

def getTestIter(fiveOrSeven, batch_size):
    assert fiveOrSeven == 5 or fiveOrSeven == 7
    testfin = codecs.open(path.join(dataPath, "qtest"+str(fiveOrSeven)), 'r', encoding = 'utf-8')
    return getDataIter(testfin, fiveOrSeven, batch_size)

def getValidIter(fiveOrSeven, batch_size):
    assert fiveOrSeven == 5 or fiveOrSeven == 7
    validfin = codecs.open(path.join(dataPath, "qvalid"+str(fiveOrSeven)), 'r', encoding = 'utf-8')
    return getDataIter(validfin, fiveOrSeven, batch_size)

def idx_to_onehot(w, vocab_size, batch_size):
    res = torch.zeros((batch_size, vocab_size)).cuda().scatter(1, w, 1)
    return torch.transpose(res,0,1)

def char_to_onehot(c, vocab_size):
    res = torch.zeros((vocab_size, 1))
    res[TEXT.vocab.stoi[c]] = 1
    return res

def sentence_to_onehot(idx, vocab_size, batch_size):
    res = torch.zeros((6*vocab_size, batch_size), dtype=torch.long)
    for i in range(batch_size):
        for j in range(6):
            res[idx[j][i]+j*vocab_size][i] = 1
    return res

def itos(idx):
    result = torch.zeros((idx.size()), dtype=torch.long)
    for a, i in enumerate(idx):
        result[a] = int(TEXT.vocab.itos[i])
    return result

def calSame(out, real):
    return int(torch.argmax(out,dim=1).eq(real).sum())

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)