from gensim.models import Word2Vec
import os
import os.path as path
import codecs
import torch
from torchtext import data
from torchtext.vocab import Vectors

wordvecPath = "word2vec.vector"
dataPath = "rnnpg_data_emnlp-2014\\partitions_in_Table_2\\rnnpg"
batch_size = 64
def tokenize(x): return x.split()

wordVec = Word2Vec.load(wordvecPath)
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
    TEXT.build_vocab(data, vectors=vectors, unk_init = torch.Tensor.normal_, min_freq=5) #构建映射,设定最低词频为5
    return torch.BucketIterator(dataset=data, batch_size=batch_size, shuffle=True)

def getTrainIter(fiveOrSeven):
    assert fiveOrSeven == 5 or fiveOrSeven == 7
    trainfin = codecs.open(path.join(dataPath, "qtrain"+fiveOrSeven), 'r', encoding = 'utf-8')
    return getDataIter(trainfin, fiveOrSeven)

def getTestIter(fiveOrSeven):
    assert fiveOrSeven == 5 or fiveOrSeven == 7
    testfin = codecs.open(path.join(dataPath, "qtest"+fiveOrSeven), 'r', encoding = 'utf-8')
    return getDataIter(testfin, fiveOrSeven)

def getValidIter(fiveOrSeven):
    assert fiveOrSeven == 5 or fiveOrSeven == 7
    validfin = codecs.open(path.join(dataPath, "qvalid"+fiveOrSeven), 'r', encoding = 'utf-8')
    return getDataIter(validfin, fiveOrSeven)
