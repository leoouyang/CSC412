from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()

import numpy as np
import os
import gzip
import struct
import array

import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve

from data import *

def sort_image_by_digit(images, labels):
    sorted = {}
    for i in range(10):
        sample_index = np.nonzero(labels[:,i])[0]
        sorted[i] = images[sample_index]
    return sorted

def find_log_likelihood_matrix(images, theta, pi):
    result = np.zeros((images.shape[0], theta.shape[0]))
    result += np.log(pi)
    for i in range(images.shape[0]):
        cur_image = images[i,:]
        for c in range(theta.shape[0]):
            cur_ll = np.dot(cur_image, np.log(theta[c,:])) + np.dot((1-cur_image), np.log((1-theta[c,:])))
            result[i,c] += cur_ll
    return result


def average_likelihood(likelihoods, label):
    log_list = likelihoods[label.nonzero()]
    return np.sum(log_list)/log_list.shape[0]

def accuracy(likelihoods,label):
    prediction= np.argmax(likelihoods,axis = 1)
    sample_index = np.array(range(likelihoods.shape[0]))
    correct = np.sum(label[sample_index,prediction])
    return correct/likelihoods.shape[0]


if __name__ == "__main__":
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    train_images = np.round(train_images[0:10000])
    train_labels = train_labels[0:10000]
    test_images = np.round(test_images[0:10000])
    test_labels = test_labels[0:10000]

    #Q1c
    sorted_images = sort_image_by_digit(train_images, train_labels)
    theta = np.zeros((10, 784))
    for c in range(theta.shape[0]):
        n_c = sorted_images[c].shape[0]
        for d in range(theta.shape[1]):
            n_cd = np.sum(sorted_images[c][:,d])
            theta[c,d] = (n_cd+1)/(n_c + 2)
    save_images(theta, "q1c.jpg")

    #Q1e
    train_log_likelihood_matrix = find_log_likelihood_matrix(train_images, theta, 0.1)
    test_log_likelihood_matrix = find_log_likelihood_matrix(test_images, theta, 0.1)

    train_average = average_likelihood(train_log_likelihood_matrix,train_labels)
    print("Training set Average log likelihood: "+ str(train_average))
    test_average = average_likelihood(test_log_likelihood_matrix,test_labels)
    print("Test set Average log likelihood: "+ str(test_average))

    train_accuracy = accuracy(train_log_likelihood_matrix, train_labels)
    print("Training set accuracy: "+ str(train_accuracy))
    test_accuracy = accuracy(test_log_likelihood_matrix, test_labels)
    print("Test set accuracy: "+ str(test_accuracy))




