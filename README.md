# RNNPG 古诗生成

本项目基于PyTorch构建了Chinese Poetry Generation with Recurrent Neural Networks (Zhang and Lapata, 2014) 中的**RNNPG**模型，实现了五言与七言绝句的生成。

代码已在**Python 3.7.4**上测试通过，所使用到的库的具体版本参见requirements.txt



文件及使用方法:

data中含有原论文提供的七言、五言绝句训练集

dataset中含有生成诗歌时使用的《诗学含英》与《平水韵》

model中含有训练(train.py)及生成(model.py)所使用的神经网络模型及训练函数，并提供了五言与七言诗已训练好的模型

shengcheng为语言模型所使用的临时目录

word2vec中含有用于生成预训练词向量的word2vec.py, 并放入了原文献提供的语料库(poemlm),训练好的词向量已放置到model文件夹内供神经网络使用