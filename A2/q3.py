from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()

import numpy as np
from scipy.misc import logsumexp

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
    # cost is the negative average sum of log probs, This is identical to cross entropy loss with softmax
    cs = y.nonzero()[1]
    sum = 0
    for i in range(x.shape[1]):
        sum = sum + log_prob(cs[i], x[:,i], w)
    average_log_prob = sum/x.shape[1]
    cost = -average_log_prob
    return cost

def softmax(z):
    #z is an 10 x 10000 matrix
    #return the matrix of probs
    return np.exp(z) / np.tile(sum(np.exp(z), 0), (len(z), 1))


def prob(x, w):
    #the input x should be 784 x 10000, the input w should be 784 x 10 in our case
    Os = np.dot(w.T, x)
    result = softmax(Os)
    return result

def cost_grad(w, x, y):
    p = prob(x, w)
    dc_do = p - y.T
    return np.dot(dc_do, x.T).T

def grad_descent(cost, cost_grad, x, y, init_w, alpha, max_iter):
    #w is 784 x 10
    w = init_w.copy()
    iter = 0
    print('Doing gradient Descent')
    cur_x = x.T #transpose x such that each column correspond to 1 sample, x is 784*10000
    while iter < max_iter:
        w -= alpha*cost_grad(w, cur_x, y)
        if iter % 20 == 0:
            print("Iter "+ str(iter))
            print("Cost: "+ str(cost(w, cur_x, y)))
        #     print "theta = ", t, ", f(x) = %.2f" % (f(x, y, t))
        #     print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    print('Final Cost is ' + format(cost(w, cur_x, y), '.2f'))
    return w

def accuracy(x, w, y):
    probs = prob(x.T, w)
    return np.mean(np.argmax(y.T, axis=0) == np.argmax(probs, axis=0))

if __name__ == "__main__":
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    train_images = np.round(train_images[0:10000])
    train_labels = train_labels[0:10000]
    test_images = np.round(test_images[0:10000])
    test_labels = test_labels[0:10000]

    w = np.zeros((784,10))
    w = grad_descent(cost, cost_grad, train_images, train_labels, w, 0.0001, 100)


    save_images(w.T, "q3c.jpg")

    print("Training set accuracy: " + str(accuracy(train_images, w, train_labels)))
    print("Training set average log prob: " + str(-cost(w, train_images.T, train_labels)))
    print("Test set accuracy: " + str(accuracy(test_images, w, test_labels)))
    print("Test set average log prob: " + str(-cost(w, test_images.T, test_labels)))

