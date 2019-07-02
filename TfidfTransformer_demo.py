from sklearn.feature_extraction.text import TfidfTransformer
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
#计算某个词出现的次数
X = vectorizer.fit_transform(corpus)

#类调用
transformer = TfidfTransformer()
# print(transformer)
#将词频矩阵统计成TF-IDF值
tfidf = transformer.fit_transform(X)
#查看数据结构tfidf[i][j]表示i类文本中tf-idf权重
print(tfidf.toarray())

'''
TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)
[[ 0.          0.43877674  0.54197657  0.43877674  0.          0.
   0.35872874  0.          0.43877674]
 [ 0.          0.24628357  0.          0.24628357  0.          0.77170162
   0.20135296  0.          0.49256715]
 [ 0.55280532  0.          0.          0.          0.55280532  0.
   0.28847675  0.55280532  0.        ]
 [ 0.          0.43877674  0.54197657  0.43877674  0.          0.
   0.35872874  0.          0.43877674]]
'''