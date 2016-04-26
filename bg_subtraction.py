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

parser = argparse.ArgumentParser(
    description="Run EM MoG model on image sequence")

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

def label_image(labels, weights, normalized_likelihood):
    max_weights_idx = np.argmax(weights, axis=0)
    max_likelihood_idx = np.argmax(normalized_likelihood, axis=0)
    diff = max_weights_idx - max_likelihood_idx
    diff[np.nonzero(diff)] = 1
    labels.append(diff)

def apply_background_subtractor(data, num_classes=3, prior_weight=3):
    # Initialize container variables with a slot for each label
    sample_img = np.array(Image.open(data[0]).getdata())
    num_pixels = sample_img.shape[0]
    num_colors = sample_img.shape[1]
    labels = []

    # Initialize mean and covariance randomly
    mean = 255 * np.random.rand(num_classes, num_pixels, num_colors)
    covariance = np.square(128 * np.random.rand(num_classes, num_pixels,
                                                num_colors, num_colors))
    initialize_covariance(covariance)
    weights = (1 / float(num_classes)) * np.ones((num_classes, num_pixels))

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
        current_img = np.array(Image.open(data[t]).getdata())
        # Get the current image and compute likelihoods
        pixels = np.array([current_img] * num_colors)
        likelihood = weights * multivariate_normal_vec(pixels, mean,
                                                       covariance, num_colors)
        normalized_likelihood = likelihood / likelihood.sum(axis=1)[:, np.newaxis]
        label_image(labels, weights, normalized_likelihood)

        # Update sufficient statistics
        N += normalized_likelihood
        M += normalized_likelihood[:, :, np.newaxis] * pixels
        Z += normalized_likelihood[:, :, np.newaxis, np.newaxis] * \
             np.einsum('ijk,ijl->ijkl', pixels, pixels)

        # Recompute mean/covariance
        weights = N / N.sum(axis=1)[:, np.newaxis]
        mean = M / N[:, :, np.newaxis]
        covariance = (1 / N[:, :, np.newaxis, np.newaxis]) * Z - np.einsum('ijk,ijl->ijkl', mean, mean)

    # Output foreground/background predictions
    for i in range(len(labels)):
        labeled_img = labels[i]
        skio.imsave('output/img_{}.png'.format(i), labeled_img.reshape((720, 1280)) * 255)

def main():
    # Import ppm files as numpy array
    data = []
    for img in os.listdir('img/'):
        # data.append(Image.open('img/' + img))
        data.append('img/' + img)
    # data = np.array([np.array(img.getdata()) for img in data])
    apply_background_subtractor(data) 

if __name__ == '__main__':
    main()
