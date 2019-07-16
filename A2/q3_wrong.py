from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam
from autograd.scipy.misc import logsumexp
from autograd.scipy.special import expit as sigmoid

import os
import gzip
import struct
import array

import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve

from data import *

def log_prob(c, x, w):
    log_numerator = np.dot(w.T[c, :], x)
    wtx = np.dot(w.T, x)
    log_denominator = logsumexp(np.array(wtx))
    return log_numerator - log_denominator


def cost(w, x, y):
    # cost is the sum of negative log porbs for
    # log(a/b) = log(a) - log(b)

    # print(np.array(w))
    # wtx = np.dot(w.T, x)
    # log_denominator = logsumexp(wtx,axis = 0)
    # w_selected = w[:,y.nonzero()[1]] #result in an 784 x 10000 weight matrix, each column
    #                                 # are the weights correspond to an sample's label
    # wx = w_selected*x
    # # log_numerator = np.sum(wx, axis = 0)
    # log_numerator = np.zeros(log_denominator.shape)
    # for i in range(log_numerator.shape[0]):
    #     sum = 0
    #     for j in range(wx.shape[0]):
    #         sum += wx[j,i]
    #     log_numerator[i] = sum
    # # print(log_numerator.shape)
    # print(log_numerator)
    # print(log_denominator)
    cs = y.nonzero()[1]
    sum = 0
    for i in range(x.shape[1]):
        sum = sum + log_prob(cs[i], x[:,i], w)
    average_log_prob = sum/x.shape[1]
    cost = -average_log_prob
    return cost

def grad_descent(cost, x, y, init_w, alpha, max_iter):
    #w is 784 x 10
    w = init_w.copy()
    iter = 0
    df = grad(cost, 0)
    print('Doing gradient Descent')
    cur_x = x.T #transpose x such that each column correspond to 1 sample, x is 784*10000
    while iter < max_iter:
        print(iter)
        print("Cost: " + str(cost(w, cur_x, y)))
        w -= alpha*df(w, cur_x, y)
        if iter % 100 == 0:
            print("Iter"+ str(iter))
            print("Cost: "+ str(cost(w, cur_x, y)))
        #     print "theta = ", t, ", f(x) = %.2f" % (f(x, y, t))
        #     print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    print('Final Cost is ' + format(cost(w, cur_x, y), '.2f'))
    return w

if __name__ == "__main__":
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    train_images = np.round(train_images[0:10000])
    train_labels = train_labels[0:10000]
    test_images = np.round(test_images[0:10000])
    test_labels = test_labels[0:10000]

    w = np.random.rand(784,10)
    w = grad_descent(cost, train_images, train_labels, w, 0.00001, 10000)
