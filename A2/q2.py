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

def top_prob_list(theta, half_image):
    #return p(x_top|c, theta, pi) for c = 0 to 9
    results = []
    for c in range(theta.shape[0]):
        cur_theta = theta[c, 0:half_image.shape[0]]
        results.append(np.prod(cur_theta**half_image * (1-cur_theta)**(1-half_image)))
    return results

if __name__ == "__main__":
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    train_images = np.round(train_images[0:10000])
    train_labels = train_labels[0:10000]
    test_images = np.round(test_images[0:10000])
    test_labels = test_labels[0:10000]

    sorted_images = sort_image_by_digit(train_images, train_labels)
    theta = np.zeros((10, 784))
    for c in range(theta.shape[0]):
        n_c = sorted_images[c].shape[0]
        for d in range(theta.shape[1]):
            n_cd = np.sum(sorted_images[c][:, d])
            theta[c, d] = (n_cd + 1) / (n_c + 2)

    # q2c
    np.random.seed(1) #make sure the output is the same every time
                        #with this seed, generated digits are:[5 8 5 0 0 1 7 6 2 4]
    num_samples = 10
    cs = np.random.randint(low=0, high=9, size=num_samples) #since pi is 0.1 for all digits, uniformly distributed
    print("generated digits are:" + str(cs))
    generated_samples = np.zeros((num_samples, 784))
    for i in range(num_samples):
        x = np.random.uniform(size=784) #generate random floats that are greater and equal to 0 and less than 1
        x = (x < theta[cs[i],:]).astype(int)    #let theta_cd = p0, a generated number is smaller the p0 with prob p0.
                                                #Therefore, if a genearated number < theta_cd, the corresponding pixel should be 1
        generated_samples[i,:] = x
    save_images(generated_samples, "q2c.jpg")

    # q2f
    num_images = 20
    result = np.zeros((20,784))
    images_from_train = train_images[0:20, :]
    result[:,0:392] = images_from_train[:,0:392]
    for im in range(num_images):
        top_probs = top_prob_list(theta, images_from_train[im,0:392])
        top_prob_sum = np.sum(top_probs)
        for i in range(392,784):
            numerator = 0.0
            for c in range(10):
                numerator += theta[c,i] * top_probs[c]
            result[im,i] = numerator/top_prob_sum
    save_images(result, "q2f.jpg")


