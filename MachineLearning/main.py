# !/usr/bin/python
# -*- coding: UTF-8 -*-

##########################
# Creator: Javy
# Create Time: 20170415
# Email: xujavy@gmail.com
# Description: Machine Learning - Chapter two
##########################

import sys
sys.path.append('/home/javy/Documents/python')

#Sample one
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()

import matplotlib.pyplot as plt
import numpy as np

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
x = df.iloc[0:100, [0, 2]].values
plt.scatter(x[0:50, 0], x[0:50, 1], color='red', marker='o', label='setosa')
plt.scatter(x[50:100, 0], x[50:100, -1], color='blue', marker='x', label='versicolor')
plt.xlabel('Petal Length')
plt.ylabel('Sepal Length')
plt.legend(loc='upper left')
plt.show()
# plt.savefig('/home/javy/Documents/DeepLearningResult/test01.png')

#Sample two
ppn = Perceptron.Perceptron(eta=0.1, n_iter=10)
ppn.fit(x, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test02.png')

#Sample Three
#import plot_description_regions
plot_description_regions.plot_description_regions(x, y, classifier=ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test03.png')

# Sample  Four
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
adal = AdalineGD.AdalineGD(n_iter=10, eta=0.01).fit(x, y)
ax[0].plot(range(1, len(adal.cost_) + 1), np.log10(adal.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')
ada2 = AdalineGD.AdalineGD(n_iter=10, eta=0.0001).fit(x, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test04.png')

#Sample Five
X_std = np.copy(x)
X_std[:,0] = (x[:,0] - x[:,0].mean()) / x[:,0].std()
X_std[:,1] = (x[:,1] - x[:,1].mean()) / x[:,1].std()

ada = AdalineGD.AdalineGD(n_iter=15, eta=0.01).fit(X_std, y)
plot_description_regions.plot_description_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('Sepal length [standardlized]')
plt.ylabel('Petal length [standardlized]')
plt.legend(loc='upper left')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test05.png')
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test06.png')

#Sample six
ada = AdalineSGD.AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
plot_description_regions.plot_description_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('Sepal length [standardlized]')
plt.ylabel('Petal length [standardlized]')
plt.legend(loc='upper left')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test07.png')
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()
#plt.savefig('/home/javy/Documents/DeepLearningResult/test08.png')
