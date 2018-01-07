# !/usr/bin/python
# -*- coding: UTF-8 -*-

##########################
# Creator: Javy
# Create Time: 20170415
# Email: xujavy@gmail.com
# Description: AdalineSGD
##########################

from numpy.random import seed
import numpy as np

class AdalineSGD(object):
    """ADAptive Linear NEuron classifier.

    Parameters
    -------------
    eta : float
        Learning rate (between 0.0 and 1.0)

    n_iter : int
        Passes over the training dataset

    Attributes
    -------------
    w_ : Id-array
        Weights after fitting
    errors_ : list
        Numbers of misclassifications in every epoch
    shuffle_state : bool (default: True)
        Shuffles training data every epoch
        if True to prevent cycles.
    random_state : int (default : None)
        Set random state for shuffling
        and initializing the weights.

    """
    def __init__(self, eta=0.01, n_iter=10,
                shuffle=True, random_state=None):
        #super(Percentron, self).__init__()
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
           seed(random_state)

    def fit(self, X, y):
        #pass
        '''Fit Training data.

        Parameters
        -----------
        x : {arrary-like}, shape = {n_samples, n_features}
            Training vectors, where n_samples
            is the number of samples and n_features is the
            number of features.
        y : arrary-like, shape = [n_samples]
            Target values.

        Return
        ----------
        self : object

        '''
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                #pass
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def  partial_fit(self, X, y):
        #pass
        ''' Fit training data without reintializing the weights '''
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        #pass
        ''' Shuffle training data'''
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        #pass
        '''initialize weights to zeros'''
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        #pass
        '''Apply Adaline learning rule to update the weights '''
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        #pass
        '''Calculate net input '''
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        #pass
        ''' Compute linear activation'''
        return self.net_input(X)

    def predict(self, X):
        #pass
        '''Return Class label after unit step'''
        return np.where(self.net_input(X) >= 0.0, 1, -1)
