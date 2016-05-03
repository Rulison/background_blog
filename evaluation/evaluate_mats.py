import argparse
import cv2
import numpy as np
from scipy.misc import imread
from scipy.io import savemat, loadmat
import string
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description="Compare predicted labels with ground truth labels.")
parser.add_argument('-p', '--predicted_mat',
                    help="Path to predicted mat made by labeled_image_to_mat.py")
parser.add_argument('-t', '--truth_mat',
                    help="Path of ground truth mat")
parser.add_argument('-m', '--mode',
                    help='Evaluation mode. Use all to compute an aggregate' +
                         'accuracy or single to compute accuracy on each frame.',
                    default='all')
parser.add_argument('-g', '--graph',
                    help='Whether or not to save plot of accuracy vs frame number.',
                    default=False)
parser.add_argument('-o', '--output',
                    help='Where to save plot.')
args = parser.parse_args()

def compare_all_mats(predicted, truth, min_shape):
    return np.sum(np.abs(predicted[:min_shape[0], :min_shape[1], :min_shape[2]] -
                          truth[:min_shape[0], :min_shape[1], :min_shape[2]]))
def compare_single_mats(predicted, truth, min_shape):
    diff = np.abs(predicted[:min_shape[0], :min_shape[1], :min_shape[2]] -
                          truth[:min_shape[0], :min_shape[1], :min_shape[2]])
    return np.sum(np.sum(diff,axis=2), axis=1)
def main():
    predicted_mat = loadmat(args.predicted_mat)['labels']
    truth_mat = loadmat(args.truth_mat)['GT']
    mode = args.mode
    
    min_shape = np.minimum(predicted_mat.shape, truth_mat.shape)
    error = None
    if(mode == 'all'):
        error = compare_all_mats(predicted_mat, truth_mat, min_shape)
        print 1 - error/(min_shape[0]*min_shape[1]*min_shape[2])
    else:
        error = compare_single_mats(predicted_mat, truth_mat, min_shape)
        error = 1 - error/(min_shape[0]*min_shape[1])
        print error
    should_graph = args.graph
    if(should_graph == 'True'):
        y_axis = error
        x_axis = np.arange(len(y_axis))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        print type(y_axis)
        #ax.plot(x_axis, y_axis)
        ax.scatter(x_axis, y_axis)
        ax.set_xlim([0,len(x_axis)])
        ax.set_ylim([0,1])
        plt.savefig(args.output)

if __name__ == '__main__':
    main()