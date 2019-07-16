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

from data import load_mnist, plot_images, save_images

# Load MNIST and Set Up Data
N_data, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = np.round(train_images[0:10000])
train_labels = train_labels[0:10000]
test_images = np.round(test_images[0:10000])


# Starter Code for 4d
# A correct solution here only requires you to correctly write the neglogprob!
# Because this setup is numerically finicky
# the default parameterization I've given should give results if neglogprob is correct.
K = 30
D = 784

# Random initialization, with set seed for easier debugging
# Try changing the weighting of the initial randomization, default 0.01
init_params = npr.RandomState(0).randn(K, D) * 0.01

# Implemented batching for you
batch_size = 10
num_batches = int(np.ceil(len(train_images) / batch_size))
def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)

# This is numerically stable code to for the log of a bernoulli density
# In particular, notice that we're keeping everything as log, and using logaddexp
# We never want to take things out of log space for stability
def bernoulli_log_density(targets, unnormalized_logprobs):
    # unnormalized_logprobs are in R
    # Targets must be 0 or 1
    t2 = targets * 2 - 1
    # Now t2 is -1 or 1, which makes the following form nice
    label_probabilities = -np.logaddexp(0, -unnormalized_logprobs*t2)
    return np.sum(label_probabilities, axis=-1)   # Sum across pixels.

def batched_loss(params, iter):
    data_idx = batch_indices(iter)
    return neglogprob(params, train_images[data_idx, :])

def neglogprob(params, data):
    # Implement this as the solution for 4c!

    pi = 1/K
    result = 0
    for i in range(data.shape[0]):
        log_prod_ber_probs = bernoulli_log_density(data[i,:], params)
        p_xc = pi * np.exp(log_prod_ber_probs)
        result = result - np.log(np.sum(p_xc))
    return result


# Get gradient of objective using autograd.
objective_grad = grad(batched_loss)

def print_perf(params, iter, gradient):
    if iter % 500 == 0:
        save_images(sigmoid(params), 'q4plot.png')
        print("Iter: " + str(iter))
        print(batched_loss(params, iter))

def top_prob_list(theta, half_image):
    #return p(x_top|c, theta, pi) for c = 0 to 9
    results = []
    for c in range(theta.shape[0]):
        cur_theta = theta[c, 0:half_image.shape[0]]
        results.append(np.prod(cur_theta**half_image * (1-cur_theta)**(1-half_image)))
    return np.array(results)


# The optimizers provided by autograd can optimize lists, tuples, or dicts of parameters.
# You may use these optimizers for Q4, but implement your own gradient descent optimizer for Q3!
optimized_params = adam(objective_grad, init_params, step_size=0.2, num_iters=10000, callback=print_perf)

num_images = 20
result = np.zeros((20, 784))
images_from_train = train_images[0:20, :]
result[:, 0:392] = images_from_train[:, 0:392]
for im in range(num_images):
    top_probs = top_prob_list(sigmoid(optimized_params), images_from_train[im, 0:392])
    top_prob_sum = np.sum(top_probs)
    for i in range(392, 784):
        numerator = 0.0
        for c in range(K):
            numerator += sigmoid(optimized_params[c, i]) * top_probs[c]
        result[im, i] = numerator / top_prob_sum
save_images(result, "q4d.jpg")
