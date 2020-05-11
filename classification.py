import codecs
from gensim.models import KeyedVectors

wordvecPath = "D:\\pku\\2020春\\智能科学引论\\团队项目\\sgns.literature.word.bz2"
sxhyPath = "D:\\pku\\2020春\\智能科学引论\\团队项目\\shixuehanying.txt"

wordVec = KeyedVectors.load_word2vec_format(wordvecPath,\
    binary = False, encoding = "utf-8", unicode_errors = "ignore")

vocab = wordVec.wv.vocab
shixuehanying = codecs.open(sxhyPath, encoding = 'UTF-8')

classes = []
for line in shixuehanying:
    l = line.split()
    if len(l) == 0 or not l[0][0].isdigit():
        continue
    for word in l[1:]:
        if word in vocab:
            classes.append(word)
            break
    else:
        classes.append(None)

""" @params keywords 是单词的可迭代对象
      @return 返回最接近的两个类的编号
"""
def get_class(keywords):
    sim = [1]*len(classes)
    for word in keywords:
        if not word in vocab:
            continue
        for i in range(len(classes)):
            if classes[i] == None:
                sim[i] = 0
                continue
            else:
                sim[i] *= wordVec.wv.similarity(classes[i], word)
    l = [(sim[i], i) for i in range(len(sim))]
    l.sort(reverse = True)
    return l[0][1], l[1][1]
            

