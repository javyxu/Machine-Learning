# !/usr/bin/python
# -*- coding: UTF-8 -*-

##########################
# Creator: Javy
# Create Time: 20170428
# Email: xujavy@gmail.com
# Description: rbf_kernel_pca
##########################

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.

    Parameters
    --------------
    X: {Numpy ndarray}, shape = [n_samples, n_features]

    gamma: float
        Tuning parameters of the RBF kernel

    n_components: int
        Number of principal components to return

    Returns
    --------------
    X_pc: {Numpy ndarray}, shape = []n_samples, k_features]
        Projected dataset

    lambdas: list
        Eigenvalues

    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distnaces into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenparis from  the centered kernel matrix
    # numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    # Collection the top k eigenvectors (Projected samples)
    alphas = np.column_stack((eigvecs[:, -i]
                            for i in range(1, n_components + 1)))

    # Collection the corresponding eigenvalues
    lambdas = [eigvals[-i] for i in range(1, n_components + 1)]

    return alphas, lambdas
    #return alphas
