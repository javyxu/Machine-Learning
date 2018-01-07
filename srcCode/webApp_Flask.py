# !/usr/bin/python
# -*- coding: UTF-8 -*-

##########################
# Creator: Javy
# Create Time: 20170510
# Email: xujavy@gmail.com
# Description: Machine Learning - Chapter nine
##########################

# Sample one
# Create tree
lst_flask_app_1/
    app.py
    templates/
        first_app.html

from flask import Flask, render_template

app =Flask(__name__)

@app.route('/')
def index():
    return render_template('first_app.html')

if __name__ == '__main__':
    app.run()

# Create tree
lst_flask_app_2/
    app.py
    static/
        style.css
    templates/
        _formhelpers.html
        first_app.html
        hello.html

from flask import flask, render_template, request
from wtforms import From, TextAreaField, validators

app = Flask(__name__)

class HelloForm(From):
    sayHello = TextAreaField('', [validators.DataRequired()])

@app.route('/')
def index():
    form = HelloForm(request.form)
    return render_template('first_app.html', form=form)

@app.route('hello', methods=['POST'])
def hello():
    form = HelloForm(request.form)
    if request.method == 'POST' and form.validate()):
        name = request.form('sayhello')
        return render_template('hello.html', name=name)
    return render_template('first_app.html', form=form)

if __name__ = '__main__':
    app.run(debug=True)


# Sample two
import pickle
import sqlite3
import numpy as np
import os

# import HashingVectorizer from local dir
from vectorizer import vect

def update_model(db_path, model, batch_size=10000):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT * FROM review_db')

    results = c.fetchmany(batch_size)
    while  results:
        data = np.array(results)
        X = data[:, 0]
        y = data[:, 1].astype(int)

        classes = np.array([0, 1])
        X_train = vect.transformat(X)
        clf.partial_fit(X_train, y, classes=classes)
        results = c.fetchmany(batch_size)

    conn.close()
    return None

cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'review_db.sqlite')
update_model(db_path=db, model=clf, batch_size=10000)
