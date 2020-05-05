from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import codecs

wordvecPath = "D:\\pku\\2020春\\智能科学引论\\团队项目\\word2vec.txt"


wordVec = KeyedVectors.load_word2vec_format(wordvecPath, binary = False)
vocab = wordVec.wv.vocab


""" params: n if number of clusters, path is the path of train file
    return: res -> {character:(label, probability)}
    using KMeans cluster
"""
def cluster(n, path):
    trainfin = codecs.open(path, 'r', encoding = 'utf-8')
    wordscnt = {}
    for line in trainfin:
        for c in line.split():
            wordscnt[c] = wordscnt.get(c, 0) + 1
    words = [c for c in wordscnt.keys() if c in vocab]
    vectors = [wordVec[c] for c in words]
    clt = KMeans(n)
    clt.fit(vectors)

    freq = [0]*n
    for i in range(len(words)):
        freq[clt.labels_[i]] += wordscnt[words[i]]
    res = {}
    for i in range(len(words)):
        c = words[i]
        label = clt.labels_[i]
        prob = wordscnt[c] / freq[label]
        res[c] = tuple((label, prob))
    for i in range(100):
        c = words[i]
        label = res[c]
        print("{} : {}".format(c, label))
    return res


