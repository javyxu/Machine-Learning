# !/usr/bin/python
# -*- coding: UTF-8 -*-

##########################
# Creator: Javy
# Create Time: 20170519
# Email: xujavy@gmail.com
# Description: Machine Learning - Chapter twelve
##########################

# Sample one
X_train, y_train = load_mnist('mnist', kind='train')
print('rows:{0}, colunms{1}'.format((X_train.shape[0]), X_train.shape[1]))

X_test, y_test = load_mnist('mnist', kind='t10k')
print('rows:{0}, colunms{1}'.format((X_test.shape[0]), X_test.shape[1]))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmp='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmp='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# Sample two
np.savetxt('train_img.csv', X_train, fmt='%i', delimiter=',')
np.savetxt('train_labels.csv', y_train, fmt='%i', delimiter=',')
np.savetxt('test_img.csv', X_test, fmt='%i', delimiter=',')
np.savetxt('test_labels.csv', y_test, fmt='%i', delimiter=',')

X_train = np.genfromtxt('train_img.csv', dtype=int, delimiter=',')
y_train = np.genfromtxt('train_labels.csv', dtype=int, delimiter=',')
y_test = np.genfromtxt('test_img.csv', dtype=int, delimiter=',')
y_test = np.genfromtxt('test_labels.csv', dtype=int, delimiter=',')

nn = NeuralNetMLP(n_output=10, n_features=X_train.shape[1], n_hidden=50,
                  l2=0.1, l1=0.0, epochs=1000, eta=0.001, decrease_const=0.000001,
                  shuffle=True, minibatches=50, random_state=1)
nn.fit(X_train, y_train, print_progress=True)

plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')
plt.tight_layout()
plt.show()

batches = np.array_split(range(len(nn.cost_)), 1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [nP.mean(cost_ary[i]) for i in batches]

plt.plot(range(len(cost_avgs)), Cost, color='red')
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()

# Sampel three
y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: {0}%%'.format(acc * 100))

y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Training accuracy: {0}%%'.format(acc * 100))

miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax =ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearst')
    ax[i].set_title('{0} t: {1} p: {2}',formt((i+1, correct_lab[i], miscl_lab[i])))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

nn_check = MLPGradientCheck(n_output=10, n_features=X_train.shape[1],
                            n_hidden=10, l2=0.0, l1=0.0, epochs=10,
                            eta=0.001, alpha=0.0, decress_const=0.0,
                            minibatches=1, random_state=1)
nn_check.fit(X_train[:5], y_train[:5], print_progress=False)
