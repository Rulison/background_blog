import argparse
import cv2
import numpy as np
from scipy.misc import imread
from scipy.io import savemat
import string
import os
import skimage.io as skio
import sys

parser = argparse.ArgumentParser(
    description="Convert labeled images to a single mat file.")
parser.add_argument('-f', '--filepath',
                    help="Path to directory containing image sequence")
parser.add_argument('-o', '--output',
                    help="Path to write output mat")
parser.add_argument('-fore', '--foreground',
	                help="Intensity of foreground pixels",
	                default=255)
parser.add_argument('-s', '--shadow',
					help="Intensity of shadow pixels",
					default=None)
args = parser.parse_args()

def convert_image_to_mat(filepath, output_mat, index, foreground_color,  shadow_color=None):

	img = imread(filepath)
	label_mat = np.zeros(shape=(img.shape[0], img.shape[1]))

	for i in range(label_mat.shape[0]):
		for j in range(label_mat.shape[1]):
			label_mat[i,j] = 1 if np.array_equal(img[i,j], np.array(foreground_color)) else 0
			if(shadow_color):
				label_mat[i,j] = 2 if np.array_equal(img[i,j], np.array(shadow_color)) else label_mat[i,j]

	output_mat[:,:,index] = label_mat

	return label_mat

def main():
	directory = args.filepath
	foreground_color = int(args.foreground)
	shadow_color = args.shadow

	output = args.output

	filenames = os.listdir(args.filepath)
	filenames = [x for x in filenames if x[0] != '.']

	img = imread(directory+'/'+filenames[0])

	output_mat = np.zeros(shape=(img.shape[0], img.shape[1], len(filenames)))

	for i in range(len(filenames)):
		filename = filenames[i]
		convert_image_to_mat(directory+'/'+filename, output_mat, i, foreground_color,  shadow_color)
	savemat(output, {'labels': output_mat})

if __name__ == '__main__':
    main()