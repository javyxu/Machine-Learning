# !/usr/bin/python
# -*- coding: UTF-8 -*-

##########################
# Creator: Javy
# Create Time: 20170520
# Email: xujavy@gmail.com
# Description: Machine Learning - Chapter thirteen
##########################

# About Theano
pip install Theano

print(theano.config.floatX)
theano.config.floatX = 'float32'
export THEANO_FLAGS=floatX=float32
THEANO_FLAGS=floatX=float32 python script.py

print(theano.config.device)
THEANO_FLAGS=device=cpu,floatX=float64 python script.py
THEANO_FLAGS=device=gpu,floatX=float64 python script.py

echo -e "\n[global]\nfloatX=float32\ndevice=gpu\n" >> ~./theanorc

pip install Keras

# Sampel one
import theano
from theano import tensor as T

# initialize
x1 = T.scalar()
w1 = T.scalar()
w0 = T.scalar()
z1 = w1 * x1 + w0

# compile
net_input = theano.function(inputs=[w1, x1, w0], outputs=z1)

# execute
print('Net input: {0}'.format(net_input(2.0, 1.0, 0.5)))


# Sample two
import numpy as np

# initialize
x = T.fmatrix(name='x')
X_sum = T.sum(x, axis=0)

# compile
clac_sum = theano.function(inputs=[x], outputs=[X_sum])

# execute (Python list)
ary = [[1, 2, 3], [1, 2, 3]]
print('Colunm sum {0}'.format(clac_sum(ary)))

# execute (Python list)
ary = np.array([[1, 2, 3], [1, 2, 3]], dtype=theano.config.floatX)
print('Colunm sum {0}'.format(clac_sum(ary)))

# Sample three
# initialize
x = T.fmatrix('x')
w= theano.shared(np.asarray([[0.0, 0.0, 0.0], dtype=theano.config.floatX))

z = x.dot(w.T)
update = [[w, w + 1.0]]

# compile
net_input = theano.function(inputs=[x], update=update, outputs=z)

# execute (Python list)
data = np.array([[1, 2, 3]], dtype=theano.config.floatX)
for i in range(5):
    print('z{0}'.format(i), net_input(data))

# Sample four
# initialize
data = np.array([1, 2, 3], dtype=theano.config.floatX)
x = T.fmatrix('x')
w= theano.shared(np.asarray([[0.0, 0.0, 0.0], dtype=theano.config.floatX))

z = x.dot(w.T)
update = [[w, w + 1.0]]

# compile
net_input = theano.function(inputs=[x], update=update, givens={x: data}, outputs=z)

# execute (Python list)
data = np.array([[1, 2, 3]], dtype=theano.config.floatX)
for i in range(5):
    print('z{0}'.format(net_input(data)))

# Sample Five
X_train = np.asarray([[0.0], [1.0],
                     [2.0], [3.0],
                     [4.0], [5.0],
                     [6.0], [7.0],
                     [8.0], [9.0]],
                     dtype=theano.config.floatX)
y_train = np.asarray([1.0, 1.3,
                      3.1, 2.0,
                      5.0, 6.3,
                      6.6, 7.4,
                      8.0, 9.0],
                      dtype=theano.config.floatX)
import theano
from theano import tensor as T
import numpy as np

def train_linreg(X_train, y_train, eta, epochs):

    costs = []
    # initialize arraya
    eta0 = T.fscalar('eta0')
    y = T.fvetor(name='y')
    X = T.fmatrix(name='X')
    w = theano.shared(np.zeros(shape=(X_train.shape[1] + 1),
                               dtype=theano.config.floatX),
                               name='w')

    # calculate cost
    net_input= T.dot(X, w[1:]) + w[0]
    errors = y - net_input
    cost = T.sum(T.pow(errors, 2))

    # preform gradient update
    gradient = T.grad(cost, wrt=w)
    update = [w, w - eta0 * gradient]

    # compile model
    train = theano.function(inputs=[eta0], output=cost,
                            updates=update, givens={X: X_train, y: y_train})

    for _ in range(epochs):
        costs.append(train(eta))

    return costs, w

def predict_linreg(X, w):
    Xt = T.matrix(name='X')
    net_input = T.dot(Xt, w[1:]) + w[0]
    predict = theano.function(inputs=[Xt], givens={w: w}, outputs=net_input)

    return predict(X)

# Sample three
import matplotlib.pyplot as plt
costs, w = train_linreg(X_train, y_train, eta=0.001, epochs=10)
plt.plot(range(1, len(costs) + 1), costs)
plt.tight_layout()
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()

plt.scatter(X_train, y_train, marker='o', s=50)
plt.plot(range(X_train.shape[0]), predict_linreg(X_train, w),
         color='gray', marker='o', markersize=4, linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Sample four
X = np.array([[1, 1.4, 1.5]])
w = np.array([0.0, 0.2, 0.4])

def net_input(X, w):
    z =X.dot(w)
    return z

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)

print('P(y=1|x) = {0}'.format(logistic_activation(X, w)[10]))

w = np.array([[1.1, 1.2, 1.3, 0.5],
              [0.1, 0.2, 0.4, 0.1],
              [0.2, 0.5, 2.1, 1.9]])

A = np.array([[1.0],
              [0.1],
              [0.3],
              [0.7]])

Z = W.dot(A)
y_probas = logistic(Z)
print('Probabilities: {0}'.format(y_probas))

y_class = np.argmax(Z, axis=0)
print('Predicted class label: {0}'.format(y_class[0]))

# Sample five
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def softmax_activation(X, w):
    z = net_input(X, w)
    return sigmoid(z)

y_probas = softmax(z)
print('Probabilities:\n {0}'.format(y_probas))
y_probas.sum()

y_class = np.argmax(Z, axis=0)
print('Predicted class label: {0}'.format(y_class[0]))

# Sample six
import matplotlib.pyplot as plt

def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)

z = np.arange(-5, 5, 0.005)
log_act = logistic(z)
tanh_act = tanh(z)

plt.ylim([-1.5, 1.5])
plt.xlabel('net input $z$')
plt.ylabel('activition $\phi(z)$')
plt.axhline(1, color='black', linestyle='--')
plt.axhline(0.5, color='black', linestyle='--')
plt.axhline(0, color='black', linestyle='--')
plt.axhline(-1, color='black', linestyle='--')

plt.plot(z, tanh_act, linewidth=2, color='black', label='tanh')
plt.plot(z, log_act, linewidth=2, color='black', label='logistic')

plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

tanh_act = np.tanh(z)

from scipy.special import expit
log_act = expit(z)

# Sample seven
import theano
theano.config,floatX = 'float32'
X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)

from keras.utils import np_utlis
print('First 3 labels: {0}'.format(y_train[:3]))
y_train_ohe = np_utlis.to_categorical(y_train)
print('\nFirst 3 labels (one-hot): {0}'.format(y_train_ohe[:3]))

# Sample eight
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizes import SGD

np.random.seed(1)

model = Sequential()
model.add(Dense(input_dim=X_train.shape[1], output_dim=50,
                init='uniform', activation='tanh'))
model.add(Dense(input_dim=50, output_dim=50,
                init='uniform', activation='tanh'))
model.add(Dense(input_dim=50, output_dim=y_train_ohe.shape[1],
                init='uniform', activation='softmax'))

sgd = SGD(lr=0.001, decay=le-7, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizes=sgd)
model.fit(X_train, y_train_ohe, nb_epoch=50, batch_size=300,
          verbose=1, validation_split=0.1, show_accuracy=True)

y_train_pred = model.predict_classes(X_train, verbose=0)
print('First 3 predictions: {0}'.format(y_train_pred[:3]))

train_cc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: {0}'.format(train_acc * 100))

y_test_pred = model.predict_classes(X_test, verbose=0)
test_acc = np.sum(y_test == y_test_std, axis=0) / X_test.shape[0]
print('Test accuracy: {0}'.format(test_acc * 100))
