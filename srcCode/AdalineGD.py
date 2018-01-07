# !/usr/bin/python
# -*- coding: UTF-8 -*-

##########################
# Creator: Javy
# Create Time：20170415
# Email: xujavy@gmail.com
# Description: AdalineGD
##########################

import numpy as np

class AdalineGD(object):
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

    """
    def __init__(self, eta=0.01, n_iter=50):
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
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def  net_input(self, X):
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
