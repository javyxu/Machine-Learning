# !/usr/bin/python
# -*- coding: UTF-8 -*-

##########################
# Creator: Javy
# Create Time: 20170510
# Email: xujavy@gmail.com
# Description: Machine Learning - Chapter nine
##########################

# Sample one
import pickle
import os

dest = os.path.join('movieclassifier', 'pcl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=4)
pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)

# Sample two
from sklearn.featue_extraction.text import HashingVectorizer
import re
import os
import pickle

cun_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl'), 'rb'))

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore', n_featues=2**21,
                        preprocessor=None, tokenizer=tokenizer)

# Sample three
import pickle
import re
import os
from vectorizer import vect
clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))

# Sample four
import numpy as np
label = {0:'negative', 1:'positive'}
example = ['I love the movie']
X = vect.transformat(example)
print('Prediction: %s\nProbability: %.2f%%' % (lable[clf.Prediction(X)[0]],
        np.max(clf.predict_proba(X)*100)))

# Sample Five
import sqlite3
import os
conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute('CREATE TABLE review_db (review TEXT, sentiment INTEGER, date TEXT)')
example1 = 'I love this movie'
c.execute('INSERT INTO review_db (review, sentiment, date) values (?, ?, DATETIME('now'))', (example1, 1))
example2 = 'I disliked this movie'
c.execute('INSERT INTO review_db (review, sentiment, date) values (?, ?, DATETIME('now'))', (example2, 0))
conn.commit()
c.execute('SELECT * FROM review_db WHERE date BETWEEN '2015-01-01 00:00:00' AND DATETIME('now')')
results = c.fetchall()
conn.close()

print(results)
