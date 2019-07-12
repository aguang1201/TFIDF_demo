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

'''
The score of MultinomialNB is 0.8348868175765646
The score of SGDClassifier is 0.9101198402130493
                        precision    recall  f1-score   support

           alt.atheism       0.95      0.80      0.87       319
         comp.graphics       0.87      0.98      0.92       389
               sci.med       0.94      0.89      0.91       396
soc.religion.christian       0.90      0.95      0.93       398

              accuracy                           0.91      1502
             macro avg       0.91      0.91      0.91      1502
          weighted avg       0.91      0.91      0.91      1502

[[256  11  16  36]
 [  4 380   3   2]
 [  5  35 353   3]
 [  5  11   4 378]]
'''