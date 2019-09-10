# chinese_nlp
A toolbox for chinese NLP project

This project is majoritly focus on Chinese text similarity.

## Popular methods:


- Simhash 方法： 数据量大，计算快
- one-hot+贝叶斯： 简单文本分类
- 主题模型: 复杂长文本
    - [math theory(LDA)](http://bloglxm.oss-cn-beijing.aliyuncs.com/lda-LDA%E6%95%B0%E5%AD%A6%E5%85%AB%E5%8D%A6.pdf)
    - [code demo](https://github.com/susanli2016/NLP-with-Python/blob/master/LDA_news_headlines.ipynb)
- 深度学习
- DSSM(Deep Structured Semantic Models)


## Plan

Apply in order:

1. Simhash 方法, becuase we have large data
2. 主题模型: because complicate text 
3. DSSM(Deep Structured Semantic Models)
4. 深度学习: use existing package/service/model


## Reference 

Recent updates: 

- 2019年3月，百度提出知识增强的语义表示模型 ERNIE（Enhanced Representation through kNowledge IntEgration），并发布了基于 PaddlePaddle 的开源代码与模型。[click here](https://github.com/PaddlePaddle/ERNIE/blob/develop/README.zh.md)

https://github.com/crownpku/Awesome-Chinese-NLP

- used for this project:
1. 工具包，代码库，深度学习数据包

https://github.com/yongzhuo/nlp_xiaojiang

- used for this project:
1. XLNET句向量-相似度（text xlnet embedding）
2. BERT句向量-相似度（Sentence Similarity）
3. 中文汉语短文本相似度


https://github.com/taozhijiang/chinese_nlp

- used for this project:
1. 主题分类
2. 贝叶斯分类

https://github.com/wainshine/Company-Names-Corpus

- used for this project:
1. 机构简称以及机构名

## Besides

1. 建立自有语料库


## Brief background knowledge

https://blog.csdn.net/u014248127/article/details/80736044 ref from this one, which is not bad

#### 1.字面距离相似度度量方法：

这一种相似性度量的方法比较简单，文本是由字词组成，重点是各种距离的度量方法。其中SimHash方法目前使用广泛。 

1. 余弦相似性：计算两个文本向量表示的余弦值。值越大越相似。 
2，简单共有词：通过计算两篇文档共有的词的总字符数除以最长文档字符数来评估他们的相似度。
3，莱文斯坦距离(编辑距离)： 
编辑距离：是指两个字串之间，由一个转成另一个所需的最少编辑操作次数（许可的编辑操作：将一个字符替换成另一个字符，插入一个字符，删除一个字符；编辑距离越小，两个串的相似度越大；动态规划计算编辑距离） 
相似度：（1-编辑距离 / 两者之间的最大长度；0-1之间）

4，Jaccard相似性系数：*Jaccard系数等于样本集交集与样本集合集的比值，即J = |A∩B| ÷ |A∪B|（交并比*）

5，欧几里得距离： 
距离计算：两个文档所有的词（不重复）在A文档的词频作为x，在B文档的作为y进行计算。距离(A,B)=平方根((x1-x2…)^2+(y1-y2….)^2) 
相似度： 1 ÷ (1 + 欧几里德距离)

6、曼哈顿距离: 
距离计算： d(i,j)=|x1-x2…|+|y1-y2…|，同理xn和yn分别代表两个文档所有的词（不重复）在A和B的词频。 
相似度： 1 ÷ (1 + 曼哈顿距离)

7，SimHash + 汉明距离： simhash是谷歌发明的算法，可以将一个文档转换成64位的字节（可以简单想象成一种hash表示文本策略）；然后我们可以通过判断两个字节的汉明距离就知道是否相似了；simhash更适用于较长的文本。 
文本的SimHash值计算过程：分词：提取文档关键词得到[word,weight]这个一个数组。（举例 [美国，4]）；hash： 用hash算法将word转为固定长度的二进制值的字符串[hash(word),weight]。（举例 [100101，4]）；加权： word的hash从左到右与权重相乘，如果为1则乘以1 ，如果是0则曾以-1。（举例4,-4,-4,4,-4,4）；合并：接着计算下个数，直到将所有分词得出的词计算完，然后将每个词第三步得出的数组中的每一个值相加。（举例美国和51区，[4,-4,-4,4,-4,4]和[5 -5 5 -5 5 5]得到[9 -9 1 -1 1 9]）；降维：对第四步得到的数组中每一个值进行判断，如果＞0记为1，如果＜0记为0。（举例[101011]）

汉明距离：两个等长字符串之间的汉明距离（Hamming distance）是两个字符串对应位置的不同字符的个数。这里就是两个SimHash值不同字符的个数。（汉明距离小于3的文本是相似的）

相似度： 1 - 汉明距离 / 最长关键词数组长度。

#### 2.语义相似性：

度量两个短文本或者说更直接的两个词语的相似性时，直接通过字面距离是无法实现的，如：中国-北京，意大利-罗马，这两个短语之间的相似距离应该是类似的，因为都是首都与国家的关系。这部分主要介绍一些词向量方法。

1，基础概念： 
统计语言模型：利用贝叶斯理论计算句子出现的概率。句子：，那么其联合概率，就是句子的概率：，通过贝叶斯理论有： 

n-gram模型：一般的模型中参数很难计算，既然每个单词依赖的单词过多，从而造成了参数过多的问题，那么我们就简单点，假设每个单词只与其前n-1个单词有关，这便是n-1阶Markov假设，也就是n-gram模型的基本思想。 
 
概率计算：通过词的评率去估计

2，One-hot词向量：用词语的出现，向量化文本 
One-hot表示：词语是否出现的0-1向量；容易造成维度灾难，并且还是不能刻画语义的信息。 
BOW模型（词袋）：词语出现的次数表示；档表示过程中并没有考虑关键词的顺序，而是仅仅将文档看成是一些关键词出现的概率的集合，每个关键词之间是相互独立的。 

TF-IDF模型：字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。词频 (term frequency, TF) 指的是某一个给定的词语在该文件中出现的次数。这个数字通常会被归一化，以防止它偏向长的文件。逆向文件频率 (inverse document frequency, IDF) 是一个词语普遍重要性的度量。某一特定词语的IDF，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取对数得到。

3，主题模型：在长文本的篇章处理中，主题模型是一种经典的模型，经常会用在自然语言处理、推荐算法等应用场景中。本节从LDA的演变过程对LDA进行阐述，然后就LDA在长文本相似性的判断聚类上做简要说明。（这个部分需要看其他文献，简单认为是文档词汇的一层低维表示） 
LSA（潜在语义分析）：一篇文档Document，词语空间的一个词频向量（每个维度表示某一词语term在该文档中出现的次数）；LSA的基本思想，便是利用最基本的SVD奇异值分解，将高维语义空间映射到低维空间；但LSA的显著问题便是只考虑词频，并不区分同一词语的不同含义。

PLSA模型： LSA基于最基本的SVD分解，但缺乏严谨的数理统计逻辑，于是Hofmann提出了PLSA，其中P便是Probabilistic，其基本的假设是每个文档所表示的词频空间向量w服从多项式分布（Multinomial distribution）；PLSA假设每篇文档的词频向量服从Categorical分布，那么对于整个训练样本的词频矩阵W则服从多项式分布。PLSA利用了aspect model，引入了潜在变量z（即所谓主题），使其变成一个混合模型（mixture model）。

LDA模型：每个文档中词的Topic分布服从Multinomial分布，其先验选取共轭先验即Dirichlet分布；每个Topic下词的分布服从Multinomial分布，其先验也同样选取共轭先验即Dirichlet分布。

4，Word2Vec模型：可以简单的认为是用神经网络学习词向量的方法；目前应用广泛，有CBOW和Skip-gram模型。 
具体介绍可以参考：word2vec原理(一) CBOW与Skip-Gram模型基础

三、语义相似度计算——DSSM：（也是基于语义的度量方法） 
DSSM（Deep Structured Semantic Models）的原理很简单，通过搜索引擎里 Query 和 Title 的海量的点击曝光日志，用 DNN 把 Query 和 Title 表达为低纬语义向量，并通过 cosine 距离来计算两个语义向量的距离，最终训练出语义相似度模型。该模型利用神经网络同时学习预测两个句子的语义相似度，又学习获得某句子的低纬语义向量表达。 
详细介绍参考：深度学习解决NLP问题：语义相似度计算——DSSM

四、结论：本文目标是如何度量两个文本之间的相似性，从字面和语义两个角度对模型算法进行整理归纳。先总结如下：对于长文本，海量数据的分析，选择SimHash方法去操作，计算快；对于简单的本文分类等问题，可以采用one-hot方法解决（进一步采用贝叶斯等方法）；对于复杂的长文本分析，可以采用主题模型；在深度学习中，对词语的表示可以采用Word2Vec模型；DSSM方法目前主要用于搜索查询问题。 
