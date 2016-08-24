import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io as skio

from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser(
    description="Plot distribution of pixel values for given x, y")
parser.add_argument('--input_dir',
    help="Directory of input files")
parser.add_argument('--num_timesteps', type=int,
    help="Number of timesteps to visualize")
args = parser.parse_args()

def viz_data(dir_name):
    while True:
        input_str = raw_input("Please input desired x and y: ")
        x, y = [int(i) for i in input_str.split()]
        data = []
        i = 0
        for filename in os.listdir(dir_name):
            img = skio.imread(os.path.join(dir_name, filename))
            data.append(img[y, x])
            if args.num_timesteps and i > args.num_timesteps:
                break
            i += 1
        data = np.array(data)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y, Z = data.T
        ax.scatter(X, Y, Z)
        ax.set_xlim([0, 255])
        ax.set_ylim([0, 255])
        ax.set_zlim([0, 255])
        fig.show()

def main():
    viz_data(args.input_dir)

if __name__ == '__main__':
    main()
