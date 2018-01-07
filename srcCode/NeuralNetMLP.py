# !/usr/bin/python
# -*- coding: UTF-8 -*-

##########################
# Creator: Javy
# Create Time: 20170520
# Email: xujavy@gmail.com
# Description: NeuralNetMLP
##########################

import numpy as np
from scipy.special import expit
import sys

class NeuralNetMLP(object):
    def __init__(self, n_output, n_features, n_hidden=30,
                 l1=0.0, l2=0.0, epochs=500, eta=0.001,
                 alpha=0.0, decress_const=0.0, shuffle=True,
                 minibatches=1, random_state=none):
        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs= epochs
        self.eta = eta
        self.alpha = alpha
        self.decress_const = decress_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _encode_labels(self, X, y):
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):
        wl = np.random.uniform(-1.0,, 1.0,
                               size=self.n_hidden*(slef.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0,, 1.0,
                               size=self.n_output*(slef.n_hidden + 1))
        w2 = w1.reshape(self.n_output, self.n_hidden + 1)
        return w1, w2

    def _sigmoid(self, z):
        # expit is equivalent to 1.0/(1.0 + np.exp(-z))
        return expit(z)

    def _add_bias_unit(self, X, how='colunm'):
        if how == 'colunm':
            X_new = np.ones((X.shape[0], X_shape[1]+1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0]+1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `colunm` or `row`')
        return X_new

    def _feedforward(self, X, w1, w2):
        a1 = self._add_bias_unit(X, how='colunm')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = a2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3

    def _L2_reg(self, lambda_, w1, w2):
        return (lambda_/2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))

    def _L1_reg(self, lambda_, w1, w2):
        return (lambda_/2.0) * (np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sun())

    def _get_cost(self, y_enc, output, w1, w2):
        term1 = -y_enc * (np.log(output))
        term2 = (1 - y_enc) * np.log(1 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, w1, w2)
        l2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + l2_term
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        # backpropagation
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w3.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma3.dot(a1)
        grad2 = sigma3.dot(a2.T)

        # regularize
        grad1[:, 1:] += (w1[:, 1:] * (self.l1 + self.l2))
        grad2[:, 1:] += (w2[:, 1:] * (self.l1 + self.l2))

        return grad1, grad2

    def predict(self, X):
        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)
        return y_pred

    def dit(self, X, y, print_progress=False):
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i range(self.epochs):

            # adpative learning rate
            self.eta /= (1 + self.decress_const*i)

            if print_progress:
                sys.stderr.write('\rEpoch:{0}/{1}'.format(i+1, self.epochs))
                sys.stderr..flush()

            if self.shuffle:
                idx = np.random.premutation(y_data.shape[0])
                X_data, y_data = X_data[idx], y_data[idx]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:

                # feedforward
                a1, z2, a2, z3, a3 = self._feedforward(X[idx], self.w1, self.w2)
                cost = self._get_const(y_enc=y_enc[:, idx], output=a3, w1=self.w1,w2=self.w2)
                self.cost_.append(cost)

                # compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2, a3=a3, z2=z2, y_enc=y_enc[:, idx],
                                                  w1=self.w1, w2=self.w2)

                # update weights
                delta_w1, delta_w2 = self.ets * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

        return self

    def _gradient_checking(self, X, y_enc, w1, w2, epsilon, grad1, grad2):
        ''' Apply gradient checking (for debugging only)

        Returns
        --------------------
        releative_error : float
           Releate error between the numerically
           approximated gradients and backpropagation gradients.

        '''

        num_grad1 = np.zeros(np.shape(w1))
        epsilon_ary1 = np.zeros(np.shape(w1))
        for i in range(w1.shape[0]):
            for j in range(w1.shape[1]):
                epsilon_ary1[i, j] = epsilon
                a1, z2, a2, z3, a3 = self._feedforward(X, w1 - epsilon_ary1, w2)
                cots1 = self._get_cost(y_enc, a3, w1 - epsilon_ary1, w2)
                a1, z2, a2, z3, a3 = self._feedforward(X, w1 + epsilon_ary1, w2)
                cost2 = self._get_cost(y_enc, a3, w1 + epsilon_ary1, w2)
                num_grad1[i, j] = (cost2 - cost1) / (2 * epsilon)
                epsilon_ary1[i, j] = 0

        num_grad2 = np.zeros(np.shape(w2))
        epsilon_ary2 = np.zeros(np.shape(w2))
        for i in range(w2.shape[0]):
            for j in range(w2.shape[1]):
                epsilon_ary2[i, j] = epsilon
                a1, z2, a2, z3, a3 = self._feedforward(X, w1, w2 - epsilon_ary2)
                cots1 = self._get_cost(y_enc, a3, w1, w2 - epsilon_ary2)
                a1, z2, a2, z3, a3 = self._feedforward(X, w1, w2 +  + epsilon_ary2)
                cost2 = self._get_cost(y_enc, a3, w1, w2 + epsilon_ary2)
                num_grad2[i, j] = (cost2 - cost1) / (2 * epsilon)
                epsilon_ary2[i, j] = 0

        num_grad = np.hstack((num_grad1.flatten(), num_grad2.flatten()))

        grad = np.hstack((grad1.flatten(), grad2.flatten()))
        norm1 = np.linalg.norm(num_grad - grad)
        norm2 = np.linalg.norm(num_grad)
        norm3 = np.linalg.norm(grad)
        releative_error = norm1 / (norm2 + norm3)
        return releative_error


class MLPGradientCheck(object):

    def fit(self, X, Y, print_progress=False):

        # Computer gradient via backpropagation
        grad1, grad2 = self._get_gradient(a1=a1, a2=a2, a3=a3,
                                          z2=z2, y_enc=y_env[:, idx],
                                          w1=self.w1, w2=self.w2)
        # start gradient checking
        grad_diff = self._gradient_checking(X=X[idx], y_enc=y_enc[:, idx],
                                            w1=self.w1, w2=self.w2, epsilon=le-5,
                                            grad1=grad1, grad2=grad2)
        if grad_diff <= le-7:
            print('OK: {0}'.format(grad_diff))
        elif grad_diff <= le-4:
            print('Warning: {0}'.format(grad_diff))
        else:
            print('PROBLEM: {0}'.format(grad_diff))

        ## end gradient checking
        # update weights: [alpha * delta_w_prev]
        # for momnetum learning
        delta_w1 = self.eta * grad1
        delta_w2 = self.eta * grad2
        self.w1 = (delta_w1 + (self.alpha * delta_w1_prev))
        self.w2 = (delta_w2 + (self.alpha * delta_w2_prev))
        delta_w1_prev= delta_w1
        delta_w2_prev = delta_w2

    return self
