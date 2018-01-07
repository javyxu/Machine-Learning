# !/usr/bin/python
# -*- coding: UTF-8 -*-

##########################
# Creator: Javy
# Create Time: 20170509
# Email: xujavy@gmail.com
# Description: Machine Learning - Chapter eight
##########################

# Sample one
import pyprind
import pandas as pd
import os
pbar = pyprind.ProgBar(50000)
labels = {'pos':1, 'neg'=0}
df = pd.DataFrame()
for s in {'test', 'train'}:
    for l in {'pos', 'neg'}:
        path = './aclImdb/%s/%s' % (s, 1)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[1]]], ignore_index=True)
            pbar.update()

df.columns = ['review', 'sentiment']

import numpy as np
np.random.seed(0)
df = df.reindex(np.random.premutation(df.index))
df.to_csv('./movie_data.csv', index=False)

df.read_csv('./movie_data.csv')
df.head(3)

# Sample two
import numpy as np
from sklearn.featue_extraction.text import CountVectorier
count = CountVectorier()
docs = np.array(['The sun is shining',
                    'The weather is sweet',
                    'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())

from sklearn.featue_extraction.text import TfidTransformer
tfidf = TfidTransformer()
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

df.loc[0, 'review'][-50]

# Sample three
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

preprocessor(df.loc[0, 'review'][-50:])
preprocessor("</a>This :) is :( a test :-)!")
df['review'] = df['review'].apply(preprocessor)

def tokenizer(text):
    return text.split()

tokenizer('runners like running and thus they run')

from nltk.stem.porter import PorterStemer
porter = PorterStemer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

tokenizer_porter('runners like running and thus they run')

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
dtop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]

# Sample four
X_train = df.loc[:25000, 'review'].values
X_text = df.loc[25000:, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
y_test = df.loc[25000:, 'sentiment'].values

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf - TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
params_grid = [{'vect__ngram_range': [(1, 1)],
                'vect__stop_words': [stop, None],
                'vect__tokenizer': [tokenizer, tokenizer_porter],
                'clf__prenalty': ['l1', 'l2'],
                'clf__C'; [1.0, 10.0, 100.0]},
                {'vect__ngram_range': [(1, 1)],
                'vect__stop_words': [stop, None],
                'vect__tokenizer': [tokenizer, tokenizer_porter],
                'vect__use_idf': [False],
                'vect__norm': [None],
                'clf__prenalty': ['l1', 'l2'],
                'clf__C'; [1.0, 10.0, 100.0]}]
lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, params_grid, scoring='accuracy',
                            cv=5, verbose=1, n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)
print('Best parameter set: %s' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.base_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

# Sample Five
import numpy as np
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with opne(path, 'r') as csv:
        next(csv) # Skip header
        for line in csv:
            text, label = line[:, -3], int(line[-2])
            yield text, lablel
next(stream_docs(path='./movie_data.csv'))

def get_minibatch(doc_stream, size):
    docs, y = [], []
        try:
            for _ in range(size):
                text, lable = next(doc_stream)
                docs.append(text)
                y.append(lable)
        except StopInteration:
            return None, None
        return docs, y

# Smaple six
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error='ignore', n_featues=2**21,
                        preprocessor=None, tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='./movie_data.csv')

import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transformat(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transformat(X_test)
print("Accuracy: %.3f" % clf.score(X_test, y_test))

clf = clf.partial_fit(X_test, y_test)
