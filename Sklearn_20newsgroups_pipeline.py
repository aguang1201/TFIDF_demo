from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics


#get the 20newsgroups dataset
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

#create the pipeline with CountVectorizer,TfidfTransformer and MultinomialNB
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                    ])
text_clf.fit(twenty_train.data, twenty_train.target)

#predict the test data with the pretrained model
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
score = np.mean(predicted == twenty_test.target)
print(f'The score of MultinomialNB is {score}')

#create the pipeline with CountVectorizer,TfidfTransformer and SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
                    ])
text_clf.fit(twenty_train.data, twenty_train.target)

#predict the test data with the pretrained model
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
score = np.mean(predicted == twenty_test.target)
print(f'The score of SGDClassifier is {score}')
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, predicted))