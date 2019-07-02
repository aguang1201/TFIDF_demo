from sklearn.feature_extraction.text import CountVectorizer

#语料
corpus = [
    'This is the first document.',
    'This is the this second second document.',
    'And the third one.',
    'Is this the first document?'
]

#将文本中的词转换成词频矩阵
vectorizer = CountVectorizer()
# vectorizer = CountVectorizer(lowercase=False)
# vectorizer = CountVectorizer(max_df=0.9, min_df=2)
print(vectorizer)
#计算某个词出现的次数
X = vectorizer.fit_transform(corpus)
print(type(X),X)
#获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
print(word)
#查看词频结果
print(X.toarray())

'''
CountVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words=None,
        strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
        tokenizer=None, vocabulary=None)
<class 'scipy.sparse.csr.csr_matrix'>   (0, 1)  1
  (0, 2)    1
  (0, 6)    1
  (0, 3)    1
  (0, 8)    1
  (1, 5)    2
  (1, 1)    1
  (1, 6)    1
  (1, 3)    1
  (1, 8)    2
  (2, 4)    1
  (2, 7)    1
  (2, 0)    1
  (2, 6)    1
  (3, 1)    1
  (3, 2)    1
  (3, 6)    1
  (3, 3)    1
  (3, 8)    1
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
[[0 1 1 1 0 0 1 0 1]
 [0 1 0 1 0 2 1 0 2]
 [1 0 0 0 1 0 1 1 0]
 [0 1 1 1 0 0 1 0 1]]
'''