# RNNPG 古诗生成

By [ZincCat](https://github.com/orgs/nlpProjectPKU/people/zinccat), [cbxg0511](https://github.com/orgs/nlpProjectPKU/people/cbxg0511), [zch65458525](https://github.com/orgs/nlpProjectPKU/people/zch65458525), [kaname-madoka18](https://github.com/orgs/nlpProjectPKU/people/kaname-madoka18)

本项目基于PyTorch构建了Chinese Poetry Generation with Recurrent Neural Networks (Zhang and Lapata, 2014) 中的**RNNPG**模型，实现了五言与七言绝句的生成。

代码已在**Python 3.7.4**上测试通过，所使用到的库的具体版本参见requirements.txt

**使用前请务必**

**在[此处](https://disk.pku.edu.cn:443/link/40BDC6EC8D361843339EE97E23606AB9)下载关键词分类所使用的词向量(由[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors) 提供，位于文学作品-Word分类下)，放置到dataset目录中。**

**在[此处](https://disk.pku.edu.cn:443/link/2021429465D38762F4920F8400AF1EE5)下载训练和预测所必须的训练集qtrain7和qtrain5文件，放置到data目录中。(注意：qtrain7与原文献提供的数据集略有差别，请勿直接使用原文献提供的qtrain7)**

**在[此处](https://disk.pku.edu.cn:443/link/7052748FBF8A0B4F15DE2395098B3C4E)下载n-gram模型，放置到根目录中。**

**在[此处](https://disk.pku.edu.cn:443/link/F5BA2C8C9969A4833024ACF92F156191)下载训练好的神经网络数据和训练所用的Word2Vec词向量，放置到model目录中。**

**如需重新训练Word2Vec，请从[此处](https://disk.pku.edu.cn:443/link/61660ED530AEA01CF2143363C8C26FAA)下载poemlm文件夹并放置到word2vec目录中。**

运行Generator.py进行绝句的生成

<img src=".\images\Poem.jpg" alt="image-20200512145410941" style="zoom: 67%;" />

运行GeneratorFixHead.py进行藏头诗的生成

<img src=".\images\FixHead.jpg" alt="image-20200512145321286" style="zoom:67%;" />

文件及使用方法:

data中含有原论文提供的七言、五言绝句训练集，注意qtrain7与RNNPG所提供的略有差别，仅使用了其前15000条，qtrain7o为原qtrain7

dataset中含有生成诗歌时使用的《诗学含英》与《平水韵》

model中含有训练及生成所使用的神经网络模型(model.py)及训练函数(train.py)，并提供了五言与七言诗已训练好的模型(.pth)

shengcheng为语言模型所使用的临时目录

word2vec中含有用于生成预训练词向量的word2vec.py, 并放入了原文献提供的语料库(poemlm),训练好的词向量已放置到model文件夹内供神经网络使用

示例：
千年纵横满尘埃，
万里明花二月开。
山色不如冰雪尽，
更看今始见春来。

月明风雨一声鸡，
自是人心在石溪。
莫道幽居无处住，
湖光秋水浸平堤。

笙歌到处游，
风作天地秋。
欲去无穷意，
悠然不识愁。
