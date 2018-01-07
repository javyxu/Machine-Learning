# !/usr/bin/python
# -*- coding: UTF-8 -*-

##########################
# Creator: Javy
# Creat Time: 20170416
# Email: xujavy@gmail.com
# Description: Machine Learning - Chapter tree
##########################

import sys
sys.path.append('/home/javy/Documents/python')

import imp
imp.reload(module)

# Sample zero
from sklearn import datasets
import matplotlib
matplotlib.Use('Agg')
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
#print('Misclassified Sample: %d' % (y_test != y_pred).sum())
print('Misclassified Sample: {0}'.format((y_test != y_pred).sum()))

from sklearn.metrics import accuracy_score
#print('Accuracy： %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy：{0}'.format(accuracy_score(y_test, y_pred)))

# Sample one
# import plot_decision_regions
plot_decision_regions.plot_decision_regions(X=X_combined_std, y=y_combined,
                                            classifier=ppn, test_idx=range(105,150))
plt.xlabel('Petal length [standardlized]')
plt.ylabel('Petal width [standardlized]')
plt.legend(loc='upper left')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test02_1.png')

#Sample two
import matplotlib.pyplot as plt
import numpy as np
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
plt.axhline(y=0.5, ls='dotted', color='k')
plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test02_2.png')

#Sample Three
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
# lr.predict_proba(X_test_std[0:1])
plot_decision_regions.plot_decision_regions(X_combined_std, y_combined, classifier=lr,
                        test_idx=range(105, 150))
plt.xlabel('Petal length [standardlized]')
plt.ylabel('Petal width [standardlized]')
plt.legend(loc='upper left')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test02_3.png')

#Sample Four
weights, params = [], []
#for c in np.arange(-5, 5):  #Integers to negative integer powers are not allowed
for c in np.arange(0, 5):
    lr = LogisticRegression(C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
plt.plot(params, weights[:, 0], label='Petal length')
plt.plot(params, weights[:, 1], linestyle='--', label='Petal width')
plt.xlabel('C')
plt.ylabel('weight cofficient')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test02_4.png')

#Sample nine
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
plot_decision_regions.plot_decision_regions(X_combined_std, y_combined, classifier=svm,
                        test_idx=range(105, 150))
plt.xlabel('Petal length [standardlized]')
plt.ylabel('Petal width [standardlized]')
plt.legend(loc='upper left')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test02_5.png')

#Sample Five
from sklearn.linear_model import SGDClassifier
ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1],
            c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1],
            c='r', marker='s', label='-1')
plt.ylim(-3.0)
plt.legend()
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test02_6.png')

#Sample six
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions.plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test02_7.png')

#Sample seven
svm = SVC(kernel='rbf', random_state=0, gamma=0.20, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions.plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.xlabel('Petal length [standardlized]')
plt.ylabel('Petal width [standardlized]')
plt.legend(loc='upper left')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test02_8.png')

#Sample eight
svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions.plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.xlabel('Petal length [standardlized]')
plt.ylabel('Petal width [standardlized]')
plt.legend(loc='upper left')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test02_9.png')

#Sample nine
import matplotlib.pyplot as plt
import numpy
def gini(p):
    return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))
def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1-p))
def error(p):
    return 1 - np.max([p, 1 - p])
x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                          ['Entropy', 'Entropy (scaled)',
                          'Gini Impurity', 'misclassification Error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgreen', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
            ncol=3, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim(0, 1.1)
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test02_10.png')

#Sample ten
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=3, random_state=0)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions.plot_decision_regions(X_combined, y_combined,
                      classifier=tree, test_idx=range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test02_11.png')

#Sample eleven
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file='tree.dot',
                feature_names=['petal length', 'petal width'])

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=10,
                                random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
plot_decision_regions.plot_decision_regions(X_combined, y_combined,
                    classifier=forest, test_idx=range(105, 150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test02_12.png')

#Sample twelve
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2,
                            metric='minkowski')
knn.fit(X_train_std, y_train)
plot_decision_regions.plot_decision_regions(X_combined_std, y_combined,
                        classifier=knn, test_idx=range(105,150))
plt.xlabel('petal length [standardlized]')
plt.ylabel('petal width [standardlized]')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test02_13.png')
