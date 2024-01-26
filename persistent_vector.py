'''
Transform metric space to finite persistent vector space through Vietoris-Rips Complex

formed a persistent vector space from metric space (X,d) based on Vietoris Rips Complex.
A vietoris Rips Complex at fixed r is obtained using the functions from vietoris_rips.py
'''

import numpy as np
from vietoris_rips import *


def euclidean_distance(xi, xj):  # compute euclidean distance between two vectors xi and xj
    return np.linalg.norm(xi-xj, ord=2)


def get_adjacency_matrix(X, **kwargs):
    # param: X is a data set
    # param: d is a distance metric function(taking two data points as parameters). default=euclidean
    # return an adjacency matrix (numpy 2d array) computed with a distance function d
    d = kwargs.get('distance_function', euclidean_distance)  # distance function of a metric space
    adjacency_matrix = np.zeros((len(X), len(X)))  # |X|*|X| matrix with dist(xi, xj) at the entry i,j
    # compute dist(xi,xj) where i != j
    for i in range(0, len(X)):
        for j in range(0, len(X)):
            adjacency_matrix[i, j] = round(d(X[i], X[j]), 7)
    return adjacency_matrix


def metric_space_to_PersVecSpace(X, k=None, **kwargs):
    # form a persistent vector space from metric space (X,d) based on Vietoris Rips Complex
    # param: X is a data set
    # param: d is a distance metric function(taking two data points as parameters). default=euclidean
    # param: k(int). Highest dimensional simplex will be k simplex for the output.
    # default k=(ex. n-1 simplex (n=# of data points))
    # return a tuple with first elem representing ascending ordered finite set of r
    # and the second elem representing persistent vector space formed my Vietoris Rips Complex at each r
    d = kwargs.get('distance_function', euclidean_distance)  # distance function of a metric space
    if k is None: k = len(X)-1
    adjacency_matrix = get_adjacency_matrix(X, d=d)
    finite_set_r = sorted(set(adjacency_matrix.flatten()))  # finite set of distances in asceding order
    pvs = []  # a list representing persistent vector space contains VR(x[r])
    for r in finite_set_r:  # iterate r in ascending order. Not iterating redundant r twice or more
        simplicial_comp = metric_space_to_VRComplex(adjacency_matrix, r=r, k=k)
        pvs.append(simplicial_comp)
    return finite_set_r, pvs