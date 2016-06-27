import argparse
import ast
import numpy as np
import os
import re
import skimage
import skimage.io as skio

parser = argparse.ArgumentParser(
    description="Read in output of BLOG program and"
                "convert to image sequence.")
parser.add_argument('--input_file',
    help="File name of input file to be parsed")
parser.add_argument('--output_dir',
    help="Name of output directory for video sequence")
parser.add_argument('--xlen', type=int,
    help="Width of input images")
parser.add_argument('--ylen', type=int,
    help="Height of input images")
args = parser.parse_args()

def parse_file():
    imgs = []
    f = open(args.input_file, 'r')
    # Strip the newline from output
    output = f.readline()[:-1]
    contents = ast.literal_eval(output)
    for query in contents:
        query_name, query_results = query
        x, y, t = query_name[6:-1].split(',')
        x, y, t = int(x[7:-1]), int(y[8:-1]), int(t[2:])
        tf_prob = {pair[0]:pair[1] for pair in query_results}
        if t > len(imgs):
            imgs.append(np.zeros((args.xlen, args.ylen)))
        if tf_prob['true'] > tf_prob['false']:
            imgs[t - 1][y, x] = 1.
        
    for i in range(len(imgs)): 
        img = imgs[i]
        skio.imsave(os.path.join(args.output_dir, 'img%d.png' % i), img)

def parse_means():
    imgs = {}
    f = open(args.input_file, 'r')
    state, timestep = None, None
    curr_val = []
    for line in f.readlines():
        contents = line.split()
        if state == 'Mean':
            if 'DiagVar' in line:
                imgs[timestep].append(curr_val)
                state = None
                curr_val = []
            else:
                curr_val.append(float(contents[0]))
        elif state is None:
            if 'TimeStep' in line:
                timestep = int(contents[2][1:])
                imgs[timestep] = []
            elif 'query' in line:
                # TODO: Add more useful indexing, etc. here
                continue
            elif 'Mean' in line:
                state = 'Mean'

    for t, vals in imgs.iteritems():
        np_array_vals = np.array(vals)
        output_img = np.reshape(np_array_vals, (args.xlen, args.ylen, 3), 'F')
        output_img = skimage.img_as_float(output_img)
        skio.imsave(os.path.join(args.output_dir, 'img%d.png' % t), output_img)


def main():
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    # parse_file()
    parse_means()

if __name__ == "__main__":
    main()
