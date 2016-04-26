import argparse
import cv2
import numpy as np
from scipy.misc import imread
from scipy.io import savemat, loadmat
import string



parser = argparse.ArgumentParser(
    description="Compare predicted labels with ground truth labels.")
parser.add_argument('-p', '--predicted_mat',
                    help="Path to predicted mat made by labeled_image_to_mat.py")
parser.add_argument('-t', '--truth_mat',
                    help="Path of ground truth mat")
args = parser.parse_args()

def compare_mats(predicted, truth, min_shape):
	return np.sum(np.abs(predicted[:min_shape[0], :min_shape[1], :min_shape[2]] -
		                  truth[:min_shape[0], :min_shape[1], :min_shape[2]]))

def main():
	predicted_mat = loadmat(args.predicted_mat)['labels']
	truth_mat = loadmat(args.truth_mat)['GT']


	min_shape = np.minimum(predicted_mat.shape, truth_mat.shape)
	error = compare_mats(predicted_mat, truth_mat, min_shape)
	print 1 - error/(min_shape[0]*min_shape[1]*min_shape[2])


if __name__ == '__main__':
    main()