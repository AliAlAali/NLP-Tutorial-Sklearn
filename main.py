from sklearn.datasets import fetch_20newsgroups
import numpy as np

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target[:10])

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform((X_train_counts))
print(X_train_tf.shape)

# training phase
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tf, twenty_train.target)
docs_new = ['God is love', "OpenGL on the GPU is fast", 'logic is a fine', 'I have vectorized this image', 'the country should seperate between religion and state']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r=> %s' % (doc, twenty_train.target_names[category]))


# easier pipeline process
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None))
])

text_clf.fit(twenty_test.data, twenty_test.target)
predicted = text_clf.predict(twenty_test.data)
print(np.mean(np.equal(predicted, twenty_test.target)))

print('\n')
# predicted = text_clf.predict(twenty_train.data)
# for doc, category in zip(twenty_train, predicted):
#     print('%r=> %s' % (doc, twenty_train.target_names[category]))

from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))