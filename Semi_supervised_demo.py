import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.semi_supervised import label_propagation
from scipy.sparse import csr_matrix
from gensim import models

tfidf = models.TfidfModel.load('my_model.tfidf')
tfidf_vec = []
for i in range(len(words)):
    string = words[i]
    string_bow = dictionary.doc2bow(string.split())
    string_tfidf = tfidf[string_bow]
    tfidf_vec.append(string_tfidf)

lsi_model = models.LsiModel(corpus = tfidf_vec,id2word = dictionary,num_topics=2)
lsi_vec = []
for i in range(len(words)):
    string = words[i]
    string_bow = dictionary.doc2bow(string.split())
    string_lsi = lsi_model[string_bow]
    lsi_vec.append(string_lsi)

data = []
rows = []
cols = []
line_count = 0
for line in lsi_vec:
    for elem in line:
        rows.append(line_count)
        cols.append(elem[0])
        data.append(elem[1])
    line_count += 1
lsi_sparse_matrix = csr_matrix((data,(rows,cols))) # 稀疏向量
lsi_matrix = lsi_sparse_matrix.toarray() # 密集向量

y = list(result.label.values)
from scipy.sparse.csgraph import *

n_total_samples = len(y)  # 1571794
n_labeled_points = 7804  # 标注好的数据共10条，只训练10个带标签的数据模型
unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]  # 未标注的数据

lp_model = label_propagation.LabelSpreading()  # 训练模型
lp_model.fit(lsi_matrix, y)

predicted_labels = lp_model.transduction_[unlabeled_indices]  # 预测的标签

# 计算被转换的标签的分布的熵
# lp_model.label_distributions_ : array,shape=[n_samples,n_classes]
# Categorical distribution for each item

pred_entropies = stats.distributions.entropy(
    lp_model.label_distributions_.T)

# 选择分类器最不确定的前2000位数字的索引
uncertainty_index = np.argsort(pred_entropies)[::1]
uncertainty_index = uncertainty_index[
                        np.in1d(uncertainty_index, unlabeled_indices)][:2000]

print(uncertainty_index)