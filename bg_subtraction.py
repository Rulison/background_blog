import argparse
import numpy as np
import os
import random
import scipy
import skimage.io as skio

from math import exp, pi
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.stats import multivariate_normal
from scipy.signal import convolve2d

parser = argparse.ArgumentParser(
    description="Run EM MoG model on image sequence")
parser.add_argument('--image_root',
    help="Directory where images to process are located")
parser.add_argument('--output',
    help="Directory to output results of processing")
parser.add_argument('--labeling', default='simple',
    help="Choose from [simple, sample] for labeling scheme")
args = parser.parse_args()

def read_images(imgpath_prefix):
    for img in os.listdir(imgpath_prefix):
        data.append(Image.open(imgpath_prefix + img))
    data = np.array([np.array(img.getdata()) for img in data])

def multivariate_normal_vec(x, mean, covariance, num_colors):
    c_inv = np.linalg.inv(covariance)
    mc = np.einsum('ijk,ijkl->ijl', (mean - x), c_inv)
    inner_terms = np.einsum('ijk,ijk->ij', mc, (mean - x))
    det = np.linalg.det(covariance)
    outer_terms = (2*pi)**(-num_colors / 2) * np.sqrt(det)
    return (1 / outer_terms) * np.exp(-0.5 * inner_terms)

def initialize_covariance(covariance):
    for i in range(covariance.shape[0]):
        for j in range(covariance.shape[1]):
            covariance[i, j] = np.diag(covariance[i, j][0])

def label_image_simple(weights, normalized_likelihood,
                       _unused_, _unused2_, num_iter=100):
    max_weights_idx = np.argmax(weights, axis=0)
    max_likelihood_idx = np.argmax(normalized_likelihood, axis=0)
    diff = max_weights_idx - max_likelihood_idx
    diff[np.nonzero(diff)] = 1
    return diff

def label_image_sample(weights, normalized_likelihood,
                       height, width, num_iter=100):
    current_labels = \
        label_image_simple(weights, normalized_likelihood, height, width)
    num_neighbors = sum_neighbors(np.ones(height*width), height, width, 3)
    max_weights_idx = np.argmax(weights, axis=0)
    unary_potentials = 1 - normalized_likelihood[max_weights_idx, 0]
    for _ in range(num_iter):
        num_fg_neighbors = sum_neighbors(current_labels, height, width, 3)
        fg_bg_diff = (num_neighbors - num_fg_neighbors) / num_neighbors
        prob_threshold = np.exp(-(unary_potentials + fg_bg_diff))
        prob_threshold /= (prob_threshold + np.exp(-(2 - unary_potentials - fg_bg_diff)))
        current_labels = np.random.random(current_labels.shape) < prob_threshold
    return current_labels

def sum_neighbors(input_vals, height, width, filter_width=3):
    reshaped_input = input_vals.reshape(height, width)
    filter_array = np.ones((filter_width, filter_width))
    convolved = convolve2d(reshaped_input, filter_array, mode='same')
    return (convolved - reshaped_input).flatten()

def save_labels(labels, height, width):
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    for i in range(len(labels)):
        labeled_img = labels[i]
        skio.imsave(os.path.join(args.output, 'img_%d.png' % i),
                    labeled_img.reshape((height, width)) * 255)

def apply_background_subtractor(data, labeling_scheme=label_image_simple,
                                num_classes=3, prior_weight=3):
    # Initialize container variables with a slot for each label
    sample_img = Image.open(data[0])
    width, height = sample_img.size[:2]
    num_pixels = width * height
    num_colors = len(sample_img.split())
    labels = []

    # Initialize mean and covariance randomly
    mean = 255 * np.random.rand(num_classes, num_pixels, num_colors)
    covariance = np.square(128 * np.random.rand(num_classes, num_pixels,
                                                num_colors, num_colors))
    initialize_covariance(covariance)
    weights = (1 / float(num_classes)) * np.ones((num_classes, num_pixels))

    # Transition matrix and prev_likelihood for basic temporal contiguity
    TRANSITION_PROB = 0.6
    prev_likelihood = weights
    transition_matrix = TRANSITION_PROB * np.eye(num_classes)
    transition_matrix[np.where(transition_matrix == 0)] = (1 - TRANSITION_PROB) / (num_classes - 1)

    # Define constants for background fraction and decay rate
    background_fraction = 0.7

    # Initialize N, M, Z based on weights, mean, covariance
    N = prior_weight * weights
    M = prior_weight * (1 / float(num_classes)) * mean
    Z = prior_weight * (1 / float(num_classes)) * \
        (covariance + np.einsum('ijk,ijl->ijkl', mean, mean))

    print("Start iteration")
    # Run background subtractor on input data
    for t in range(len(data)):
        print(t)
        current_img = np.array(Image.open(data[t]).getdata()).reshape((num_pixels, num_colors))
        # Get the current image and compute likelihoods
        pixels = np.array([current_img] * num_classes)
        likelihood = weights * multivariate_normal_vec(pixels, mean,
                                                       covariance, num_colors)
        # Apply simple Markov Model for temporal contiguity
        likelihood *= transition_matrix.dot(prev_likelihood)
        normalized_likelihood = likelihood / likelihood.sum(axis=0)[np.newaxis, :]
        prev_likelihood = normalized_likelihood
        labels.append(labeling_scheme(weights, normalized_likelihood, height, width))

        # Update sufficient statistics
        N += normalized_likelihood
        M += normalized_likelihood[:, :, np.newaxis] * pixels
        Z += normalized_likelihood[:, :, np.newaxis, np.newaxis] * \
             np.einsum('ijk,ijl->ijkl', pixels, pixels)

        # Recompute mean/covariance
        weights = N / N.sum(axis=0)[np.newaxis, :]
        mean = M / N[:, :, np.newaxis]
        covariance = (1 / N[:, :, np.newaxis, np.newaxis]) * Z - np.einsum('ijk,ijl->ijkl', mean, mean)

    # Output foreground/background predictions
    save_labels(labels, height, width)

def main():
    labeling_scheme = None
    if args.labeling == 'simple':
        labeling_scheme = label_image_simple
    elif args.labeling == 'sample':
        labeling_scheme = label_image_sample
    # Import ppm files as numpy array
    data = []
    for img in os.listdir(args.image_root):
        data.append(os.path.join(args.image_root, img))
    apply_background_subtractor(data, labeling_scheme)

if __name__ == '__main__':
    main()
