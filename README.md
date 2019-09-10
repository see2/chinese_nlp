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

1. 余弦相似性
2. 简单共有词
3. 莱文斯坦距离(编辑距离)
4. Jaccard相似性系数：*Jaccard系数等于样本集交集与样本集合集的比值，即J = |A∩B| ÷ |A∪B|（交并比*）
5. 欧几里得距离
6. 曼哈顿距离
7. SimHash + 汉明距离

#### 2.语义相似性：词向量方法。

1. n-gram模型
2. One-hot词向量
3. TF-IDF模型

#### 3.语义相似性：主题模型
1. LSA（潜在语义分析）
2. PLSA模型
3. LDA模型
4. Word2Vec模型

#### 4.DSSM

深度学习模型训练
