from sklearn.feature_extraction.text import TfidfVectorizer

# 计算TF-IDF

# 读取分词后的文本
with open('./nlp_test1.txt') as f1:
    res1 = f1.read()
with open('./nlp_test3.txt') as f2:
    res2 = f2.read()

stpwrdlst = ['the', 'an']
corpus = [res1, res2]  # 构造语料
vector = TfidfVectorizer(stop_words=stpwrdlst)  # 传递停用词代码
tfidf = vector.fit_transform(corpus)  # 得到结果

wordlist = vector.get_feature_names()  # 获取词袋模型中的所有词
# tf-idf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
weightlist = tfidf.toarray()
# 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
for i in range(len(weightlist)):
    print("-------第", i, "段文本的词语tf-idf权重------")
    for j in range(len(wordlist)):
        print(wordlist[j], weightlist[i][j])
