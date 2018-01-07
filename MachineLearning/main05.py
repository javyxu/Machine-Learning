# !/usr/bin/python
# -*- coding: UTF-8 -*-

##########################
# Creator: Javy
# Create Time: 20170501
# Emailß: xujavy@gmail.com
# Description: Machine Learning - Chapter seven
##########################

# Sample one
from scipy.misc import comb
import math
def ensembe_error(n_classifier, error):
    k_start = math.ceil(n_classifier / 2.0)
    probs = [comb(n_classifier, k) * error**k *
            (1-error)**(n_classifier - k)
            for k in range(k_start, n_classifier + 1)]
    return sum(probs)
ensembe_error(n_classifier=11, error=0.25)

# Sample two
import numpy as np
error_range = np.arange(0.0, 1.01, 0.01)
ens_error = [ensembe_error(n_classifier=11, error=error)
            for error in error_range]
import matplotlib.pyplot as plt
plt.plot(error_range, ens_error,
        label='Ensemble error', linewidth=2)
plt.plot(error_range, error_range,
        linestyle='--', label='Base error', linewidth=2)
plt.xlabel('Base error')
ply.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid()
plt.show()

# Sample three
import numpy as np
np.argmax(np.bincount([0, 0, 1],
            weights=[0.2, 0.2, 0.6]))
ex = np.array([[0.9, 0.1],
                [0.8, 0.2],
                [0.4, 0.6]])
p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])

# Sample four
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5.0, random_state=1)

# Sample five
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipline import Pipline
import numpy as np

clf1 = LogisticRegression(penatly='12', C=0.001, random_state=0)
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
pipe1 = Pipline([['sc', StandardScaler()],
                    ['clf', clf1]])
pipe3 = Pipline([['sc', StandardScaler()],
                    ['clf', clf3]])
clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
print('10-kfold cross validation:\n')
for clf, label in zip([pipe1, clf2, clf_labels]):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print('FOR AUC: %0.2f (+/- %0.2f) [%s]' %
            (scores.mean(), scores.std(), label))

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels += ['Majority Voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print('FOR AUC: %0.2f (+/- %0.2f) [%s]' %
            (scores.mean(), scores.std(), label))

# Sample six
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
colors = ['balck', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, lable, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    # asssuming the label of positive class is 1
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, t=tpr)
    plt.plot(fpr, tpr, colors=clr, linestyles=ls, lablel='%s (auc = %0.2f)' % (label, roc_auc))
plt.legend(loc='lower right')
plt.plot([0, 1, [0, 1], linestyles='--', color='gray', linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylable('True Positive Rate')
plt.show()

# Sample seven
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
from itertolls import product
x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std{:, 1}.min() - 1
y_max = X_train_std[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                    np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(7, 5))
for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z =Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0],
                                    X_train_std[y_train==0, 1],
                                    c='blue', marker='^', s=50)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==1, 0],
                                    X_train_std[y_train==1, 1],
                                    c='red', marker='o', s=50)
    axarr[idx[0], idx[1]].set_title(tt)
plt.text(-3.5, -4.5, s='Sepal width [standardlized]', ha='center', va='center', fontsize=12)
plt.text(-10.5, 4.5, s='Petal length [standardlized]', ha='center', va='center', fontsize=12, rotation=90)
plt.show()

# Sample eight
mv_clf.get_params()
from sklearn.grid_search import GridSearchCV
params = {'decisontreeclassifier__max_depth': [1, 2],
            'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator=mv_clf, params_grid=params, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)
for params, mean_score, scores in grid.grid_scores_:
    print('%0.3f+/-%0.2f %r' % (mean_score, scores.std() / 2, params))

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)


# Sample nine
import pandas as pd
df_wine = pd,.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol',
                                'Malic acid', 'Ash','Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                                'color intensity', 'Hue',
                                'OD280/OD315 of diluted wines', 'Proline']
df_wine = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label'].values
X = df_wine[['Alcohol', 'Hue']].values

# Sample ten
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=1)

from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
bag = BaggingClassifier(base_estimator=tree, n_estimators=500,
                        max_samples=1.0, max_features=1.0,
                        bootstrap=True, bootstrap_features=False,
                        n_jobs=1, random_state=1)

from sklearn.metrics import accuracy_score
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Descision tree train/tree accuracies %.3f/%.3f'
        % (tree_train, tree_test))

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(y_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print('Bagging train/tree accuracies %.3f/%.3f'
        % (bag_train, bag_test))

# Sample eleven
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = y_train[:, 1].min() - 1
y_max = y_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                    np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2,
                        sharex='col', sharey='row',figsize=(8, 3))
for idx, clf, tt in zip([0, 1], [tree, bag],
                        ['Decision Tree', 'Bagging']):
    clf.fit(X_train, y_train)

    Z = clf.predict(np.C_[xx.ravel(), yy.ravel()])
    X = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0],
                        X_train[y_train==0, 1],
                        c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train==1, 0],
                        X_train[y_train==1, 1],
                        c='red', marker='o')
    axarr[idx].set_title(tt)

axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.text(10.2, -1.2, s='Hue', ha='center', va='center', fontsize=12)
plt.show()

# Sample twelve
from sklearn.ensemble import AdaBoostClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
ada = AdaBoostClassifier(base_estimator=tree, n_estimators=500,
                        learning_rate=0.1, random_state=0)
tree = tree_fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(y_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Descision tree train/tree accuracies %.3f/%.3f'
        % (tree_train, tree_test))

ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(y_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print('AdaBoost train/tree accuracies %.3f/%.3f'
        % (tree_train, tree_test))

# Sample Thirteen
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = y_train[:, 1].min() - 1
y_max = y_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                    np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2,
                        sharex='col', sharey='row',figsize=(8, 3))
for idx, clf, tt in zip([0, 1], [tree, ada],
                        ['Decision Tree', 'AdaBoost']):
    clf.fit(X_train, y_train)

    Z = clf.predict(np.C_[xx.ravel(), yy.ravel()])
    X = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0],
                        X_train[y_train==0, 1],
                        c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train==1, 0],
                        X_train[y_train==1, 1],
                        c='red', marker='o')
    axarr[idx].set_title(tt)

axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.text(10.2, -1.2, s='Hue', ha='center', va='center', fontsize=12)
plt.show()
