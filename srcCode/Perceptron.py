# !/usr/bin/python
# -*- coding: UTF-8 -*-

##########################
# Creator: Javy
# Create Time: 20170415
# Email: xujavy@gmail.com
# Description: Perceptron
##########################

import numpy as np

class Perceptron(object):
    """Percentron classifier.
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

    """
    def __init__(self, eta=0.01, n_iter=10):
        #super(Percentron, self).__init__()
        self.eta = eta
        self.n_iter = n_iter

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
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                #pass
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def  net_input(self, X):
        #pass
        '''Calculate net input '''
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        #pass
        '''Return Class label after unit step'''
        return np.where(self.net_input(X) >= 0.0, 1, -1)
