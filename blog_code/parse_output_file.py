import argparse
import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import skimage
import skimage.io as skio

from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

parser = argparse.ArgumentParser(
    description="Read in output of BLOG program and"
                "convert to image sequence.")
parser.add_argument('--input_file',
    help="File name of input file to be parsed")
parser.add_argument('--output_dir',
    help="Name of output directory for video sequence")
parser.add_argument('--query_type',
    help="Name of desired query")
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
        output_img /= 255.
        skio.imsave(os.path.join(args.output_dir, 'img%d.png' % t), output_img)

def parse_mean_var_offline():
    imgs = []
    cov_mat = []
    f = open(args.input_file, 'r')
    state, x, y = None, None, None
    re_com = re.compile('Component\[([0-9]+)\]')
    re_imx = re.compile('ImageX\[([0-9]+)\]')
    re_imy = re.compile('ImageY\[([0-9]+)\]')
    curr_val = []
    for line in f.readlines():
        contents = line.split()
        if 'query' in line:
            comp = int(re_com.findall(line)[0])
            x = int(re_imx.findall(line)[0])
            y = int(re_imy.findall(line)[0])
            if 'Mean' in line:
                state = 'mean'
            elif 'Variance' in line:
                state = 'var'
            if comp >= len(imgs):
                for i in range(len(imgs), comp + 1):
                    imgs.append(np.zeros((args.xlen, args.ylen, 3)))
                    cov_mat.append(np.zeros((args.xlen, args.ylen, 3, 3)))
        elif 'DiagVar' in line:
            if state == 'mean':
                imgs[comp][y, x] = np.array(curr_val)
            elif state == 'var':
                cov_mat[comp][y, x] = np.array(curr_val)
            curr_val = []
            state = None
        elif ('loopend' in line) or ('Mean' in line) or state == None:
            continue
        else:
            if state == 'mean':
                pixel_val = float(line.split()[0])
                curr_val.append(pixel_val)
            elif state == 'var':
                pixel_vals = [float(i) for i in line.split()]
                curr_val.append(pixel_vals)

    for i in range(len(imgs)):
        img = imgs[i]
        skio.imsave(os.path.join(args.output_dir, 'img%d.png' % i), img / 255.)
        np.save(os.path.join(args.output_dir, 'mean%d.npy' % i), img)
        np.save(os.path.join(args.output_dir, 'var%d.npy' % i), cov_mat[i])
    colors = ['r', 'g', 'b', 'y', 'm']

    while True:
        input_str = raw_input("Please input desired x and y: ")
        x, y = [int(i) for i in input_str.split()]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for c in range(len(imgs)):
            mean, cov = imgs[c][y, x], cov_mat[c][y, x]
            X, Y, Z = np.random.multivariate_normal(mean, cov, 200).T
            ax.scatter(X, Y, Z, c=colors[c])
        ax.set_xlim([0, 255])
        ax.set_ylim([0, 255])
        ax.set_zlim([0, 255])
        fig.show()


def parse_static():
    static = np.zeros((args.xlen, args.ylen))
    f = open(args.input_file, 'r')
    state, x, y = None, None, None
    re_imx = re.compile('ImageX\[([0-9]+)\]')
    re_imy = re.compile('ImageY\[([0-9]+)\]')
    curr_val = []
    for line in f.readlines():
        contents = line.split()
        if 'query' in line:
            x = int(re_imx.findall(line)[0])
            y = int(re_imy.findall(line)[0])
        elif 'loopend' in line:
            continue
        else:
            if line.split()[0] == 'true' and float(line.split()[2]) == 1.0:
                static[y, x] = 1
    skio.imsave(os.path.join(args.output_dir, 'static.png'), static)
    np.savetxt(os.path.join(args.output_dir, 'static.txt'), static)


def parse_offline_sequence():
    imgs = []
    f = open(args.input_file, 'r')
    state, x, y, timestep = None, None, None, -1
    curr_val = {}
    re_com = re.compile('Component\[([0-9]+)\]')
    re_imx = re.compile('ImageX\[([0-9]+)\]')
    re_imy = re.compile('ImageY\[([0-9]+)\]')
    re_tim = re.compile('Time\[([0-9]+)\]')
    for line in f.readlines():
        contents = line.split()
        if 'query' in line:
            if len(curr_val) > 0:
                curr_img = imgs[t]
                if (max([(curr_val[i], i) for i in curr_val])[1] == 'Component[2]'):
                    curr_img[y, x] = 1.0
                else:
                    curr_img[y, x] = 0.0
            # x, y = int(line[line.find('ImageX[') + 7]), int(line[line.find('ImageY[') + 7])
            x = int(re_imx.findall(line)[0])
            y = int(re_imy.findall(line)[0])
            # t = int(line[line.find('Time[') + 5:-3])
            t = int(re_tim.findall(line)[0])
            curr_val = {}
            if (t > timestep):
                imgs.append(np.zeros((args.ylen, args.xlen)))
                timestep = t
        elif 'loopend' in line:
            if len(curr_val) > 0:
                curr_img = imgs[t]
                if (max([(curr_val[i], i) for i in curr_val])[1] == 'Component[2]'):
                    curr_img[y, x] = 1.0
                else:
                    curr_img[y, x] = 0.0
            curr_val = {}
        else:
            curr_val[contents[0]] = float(contents[2])

    for i in range(len(imgs)):
        img = skimage.img_as_ubyte(imgs[i])
        skio.imsave(os.path.join(args.output_dir, 'img%d.png' % i), img)

def parse_online_sequence():
    imgs = []
    f = open(args.input_file, 'r')
    state, x, y, timestep = None, None, None, 0
    curr_val = {}
    re_com = re.compile('Component\[([0-9]+)\]')
    re_imx = re.compile('ImageX\[([0-9]+)\]')
    re_imy = re.compile('ImageY\[([0-9]+)\]')
    re_tim = re.compile('@([0-9]+)\)')
    for line in f.readlines():
        contents = line.split()
        if 'query' in line:
            if len(curr_val) > 0:
                curr_img = imgs[t - 1]
                if (max([(curr_val[i], i) for i in curr_val])[1] == 'Component[2]'):
                    curr_img[y, x] = 1.0
                else:
                    curr_img[y, x] = 0.0
            # x, y = int(line[line.find('ImageX[') + 7]), int(line[line.find('ImageY[') + 7])
            x = int(re_imx.findall(line)[0])
            y = int(re_imy.findall(line)[0])
            # t = int(line[line.find('@') + 1:-2])
            t = int(re_tim.findall(line)[0])
            curr_val = {}
            if (t > timestep):
                imgs.append(np.zeros((args.ylen, args.xlen)))
                timestep = t
        elif 'loopend' in line:
            if len(curr_val) > 0:
                curr_img = imgs[t - 1]
                if (max([(curr_val[i], i) for i in curr_val])[1] == 'Component[2]'):
                    curr_img[y, x] = 1.0
                else:
                    curr_img[y, x] = 0.0
        elif 'TimeStep' not in line:
            curr_val[contents[0]] = float(contents[2])

    for i in range(len(imgs)):
        img = imgs[i]
        skio.imsave(os.path.join(args.output_dir, 'img%d.png' % i), img)


def parse_mean_particles():
    imgs = []
    f = open(args.input_file, 'r')
    state, x, y = None, None, None
    re_com = re.compile('Component\[([0-9]+)\]')
    re_imx = re.compile('ImageX\[([0-9]+)\]')
    re_imy = re.compile('ImageY\[([0-9]+)\]')
    re_val = re.compile('\[([0-9\.\-E]+)\]')
    for line in f.readlines():
        if 'Distribution' in line:
            comp = int(re_com.findall(line)[0])
            x = int(re_imx.findall(line)[0])
            y = int(re_imy.findall(line)[0])
            if comp >= len(imgs):
                for i in range(len(imgs), comp + 1):
                    imgs.append(np.zeros((args.xlen, args.ylen, 3)))
        elif 'loopend' in line:
            break
        else:
            pixel_vals = [float(i) for i in re_val.findall(line)]
            weight = float(line.split()[3])
            imgs[comp][y, x, :] += np.array(pixel_vals) * weight

    for i in range(len(imgs)):
        img = imgs[i] / 255.
        skio.imsave(os.path.join(args.output_dir, 'img%d.png' % i), img)

def parse_mean_var_particles():
    imgs = []
    cov_mat = []
    f = open(args.input_file, 'r')
    state, x, y = None, None, None
    re_com = re.compile('Component\[([0-9]+)\]')
    re_imx = re.compile('ImageX\[([0-9]+)\]')
    re_imy = re.compile('ImageY\[([0-9]+)\]')
    re_mean = re.compile('\[([0-9\.\-E]+)\]')
    re_var = re.compile('\[([0-9\.\-E]+), ([0-9\.\-E]+), ([0-9\.\-E]+)\]')
    for line in f.readlines():
        if 'Distribution' in line:
            comp = int(re_com.findall(line)[0])
            x = int(re_imx.findall(line)[0])
            y = int(re_imy.findall(line)[0])
            if 'Mean'in line:
                state = 'mean'    
            elif 'Variance' in line:
                state = 'var'
            if comp >= len(imgs):
                for i in range(len(imgs), comp + 1):
                    imgs.append(np.zeros((args.xlen, args.ylen, 3)))
                    cov_mat.append(np.zeros((args.xlen, args.ylen, 3, 3)))
        elif 'loopend' in line:
            break
        else:
            if state == 'mean':
                pixel_vals = np.array([float(i) for i in re_mean.findall(line)])
                weight = float(line.split()[3])
                imgs[comp][y, x, :] += (pixel_vals) * weight
            elif state == 'var':
                cov_vals = np.array(
                    [[float(i) for i in a] for a in re_var.findall(line)])
                weight = float(line.split()[9])
                cov_mat[comp][y, x, :] += np.reshape(cov_vals, (3, 3)) * weight

    for i in range(len(imgs)):
        img = imgs[i]
        skio.imsave(os.path.join(args.output_dir, 'img%d.png' % i), img / 255.)
        np.save(os.path.join(args.output_dir, 'mean%d.npy' % i), img)
        np.save(os.path.join(args.output_dir, 'var%d.npy' % i), cov_mat[i])

    # fig, axes = plt.subplots(args.ylen, args.xlen,
    #                          subplot_kw=dict(projection='3d'))
    colors = ['r', 'g', 'b']
    # for j in range(args.ylen):
    #     for i in range(args.xlen):
    #         for c in range(3):
    #             mean, cov = imgs[c][j, i], cov_mat[c][j, i]
    #             X, Y, Z = np.random.multivariate_normal(mean, cov, 200).T
    #             axes[j][i].scatter(X, Y, Z, c=colors[c])
    #         axes[j][i].set_xlim([0, 255])
    #         axes[j][i].set_ylim([0, 255])
    #         axes[j][i].set_zlim([0, 255])
    # plt.show()
    # plt.pause(10000)
    while True:
        input_str = raw_input("Please input desired x and y: ")
        x, y = [int(i) for i in input_str.split()]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for c in range(3):
            mean, cov = imgs[c][y, x], cov_mat[c][y, x]
            X, Y, Z = np.random.multivariate_normal(mean, cov, 200).T
            ax.scatter(X, Y, Z, c=colors[c])
        ax.set_xlim([0, 255])
        ax.set_ylim([0, 255])
        ax.set_zlim([0, 255])
        fig.show()


def parse_dblog_online():
    imgs = []
    f = open(args.input_file, 'r')
    x, y, timestep = None, None, 0
    re_imx = re.compile('ImageX\[([0-9]+)\]')
    re_imy = re.compile('ImageY\[([0-9]+)\]')
    curr_val = {}
    for line in f.readlines():
        if 'Query' in line:
            timestep += 1
            imgs.append(np.zeros((args.xlen, args.ylen)))
        elif 'Distribution' in line:
            x = int(re_imx.findall(line)[0])
            y = int(re_imy.findall(line)[0])
            curr_val = {}
        elif 'Number of samples' in line:
            continue
        else:
            comp, prob = line.split()        
            curr_val[comp] = float(prob)
            if comp == '2' and max(curr_val, key=lambda i: curr_val[i]) == '2':
                imgs[timestep - 1][y, x] = 1.

    for i in range(len(imgs)):
        img = imgs[i]
        skio.imsave(os.path.join(args.output_dir, 'img%d.png' % i), img)


def main():
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    queries = {'file': parse_file, 'means': parse_means,
               'mean_var_offline': parse_mean_var_offline,
               'offline_sequence': parse_offline_sequence,
               'online_sequence': parse_online_sequence,
               'mean_particles': parse_mean_particles,
               'mean_var_particles': parse_mean_var_particles,
               'static': parse_static,
               'dblog_online': parse_dblog_online}
    if args.query_type in queries:
        queries[args.query_type]()
    else:
        print("Query %s not in set of supported queries" % args.query_type)

if __name__ == "__main__":
    main()
