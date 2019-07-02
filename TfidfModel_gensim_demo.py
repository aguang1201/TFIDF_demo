from gensim import corpora
from gensim import models

#语料
corpus = [
        'this is the first document',
        'this is the second second document',
        'and the third one',
        'is this the first document'
    ]

word_list = []
for i in range(len(corpus)):
    word_list.append(corpus[i].split(' '))
print(word_list)
'''
    [输出]:
    [['this', 'is', 'the', 'first', 'document'],
     ['this', 'is', 'the', 'second', 'second', 'document'],
     ['and', 'the', 'third', 'one'],
     ['is', 'this', 'the', 'first', 'document']]
'''


# 赋给语料库中每个词(不重复的词)一个整数id
dictionary = corpora.Dictionary(word_list)
new_corpus = [dictionary.doc2bow(text) for text in word_list]
print(new_corpus)
# 元组中第一个元素是词语在词典中对应的id，第二个元素是词语在文档中出现的次数
'''
[输出]：
[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)], 
 [(0, 1), (2, 1), (3, 1), (4, 1), (5, 2)], 
 [(3, 1), (6, 1), (7, 1), (8, 1)], 
 [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]]
'''
 # 每个词对应的id
print(dictionary.token2id)
'''
 [输出]：
 {'document': 0, 'first': 1, 'is': 2, 'the': 3, 'this': 4, 'second': 5, 'and': 6,
 'one': 7,   'third': 8}
'''

# 训练模型并保存
tfidf = models.TfidfModel(new_corpus)
tfidf.save("my_model.tfidf")

# 载入模型
tfidf = models.TfidfModel.load("my_model.tfidf")
# 使用这个训练好的模型得到单词的tfidf值
tfidf_vec = []
for i in range(len(corpus)):
    string = corpus[i]
    string_bow = dictionary.doc2bow(string.lower().split())
    string_tfidf = tfidf[string_bow]
    tfidf_vec.append(string_tfidf)
print(tfidf_vec)

'''
[输出]：
    [[(0, 0.33699829595119235),
      (1, 0.8119707171924228),
      (2, 0.33699829595119235),
      (4, 0.33699829595119235)],
     [(0, 0.10212329019650272),
      (2, 0.10212329019650272),
      (4, 0.10212329019650272),
      (5, 0.9842319344536239)],
     [(6, 0.5773502691896258), (7, 0.5773502691896258), (8, 0.5773502691896258)],
     [(0, 0.33699829595119235),
      (1, 0.8119707171924228),
      (2, 0.33699829595119235),
      (4, 0.33699829595119235)]]
'''