from sklearn.feature_extraction.text import TfidfVectorizer


#语料
corpus = [
        'this is the first document',
        'this is the second second document',
        'and the third one',
        'is this the first document'
    ]

tfidf_vec = TfidfVectorizer()
# tfidf_vec = TfidfVectorizer(stop_words=['the', 'an'])
# tfidf_vec = TfidfVectorizer(lowercase=False)
# tfidf_vec = TfidfVectorizer(max_df=0.9, min_df=2)
print(tfidf_vec)
tfidf_matrix = tfidf_vec.fit_transform(corpus)
# 得到语料库所有不重复的词
print(tfidf_vec.get_feature_names())
# 得到每个单词对应的id值
print(tfidf_vec.vocabulary_)
# 得到每个句子所对应的向量
# 向量里数字的顺序是按照词语的id顺序来的
print(tfidf_matrix.toarray())

'''
[输出]:
    ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
    {'this': 8, 'is': 3, 'the': 6, 'first': 2, 'document': 1, 'second': 5, 'and': 0, 'third': 7, 'one': 4}
    [[0.         0.43877674 0.54197657 0.43877674 0.         0.
      0.35872874 0.         0.43877674]
     [0.         0.27230147 0.         0.27230147 0.         0.85322574
      0.22262429 0.         0.27230147]
     [0.55280532 0.         0.         0.         0.55280532 0.
      0.28847675 0.55280532 0.        ]
     [0.         0.43877674 0.54197657 0.43877674 0.         0.
      0.35872874 0.         0.43877674]]
'''