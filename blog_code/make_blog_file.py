import argparse
import numpy as np
import os
import shutil
import skimage.io as skio

parser = argparse.ArgumentParser(
    description="Read frames from input file and generate BLOG file"
                "for background subtraction.") 
parser.add_argument('--image_root',
    help="Directory where images to process are located")
parser.add_argument('--num_timesteps', type=int, default=10,
    help="Number of timesteps to read (default: 10)")
parser.add_argument('--output_name', default='output_blog.dblog',
    help="Name of output BLOG file (default: output_blog.dblog)")
args = parser.parse_args()

def read_img_intensity(img, output_file, t):
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            r, g, b = img[i, j].tolist()
            obs = 'obs Intensity(ImageX[{0}], ImageY[{1}], @{2}) = ' \
                  '[{3}; {4}; {5}];\n'.format(i, j, t, r, g, b)
            output_file.write(obs)

def enforce_spatial_constraint(img, output_file, t):
    height, width = img.shape[:2]
    obs_template = 'obs Output(ImageX[{0}], ImageY[{1}], ' \
                   'ImageX[{2}], ImageY[{3}], @{4}) = true;\n'
    for i in range(height):
        for j in range(width):
            if i < height - 1:
                obs = obs_template.format(i, j, i + 1, j, t)
                output_file.write(obs)
            if j < width - 1:
                obs = obs_template.format(i, j, i, j + 1, t)
                output_file.write(obs)

def query_label(img, output_file, t):
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            label = 'query Label(ImageX[{0}], ImageY[{1}], @{2});\n' \
                         .format(i, j, t)
            output_file.write(label)

def main():
    shutil.copyfile('bsub.dblog', args.output_name)
    output_file = open(args.output_name, 'a')
    img_filenames = os.listdir(args.image_root)

    for t in range(min(len(img_filenames), args.num_timesteps)):
        img_name = img_filenames[t]
        img = skio.imread(os.path.join(args.image_root, img_name))[:5, :5]
        read_img_intensity(img, output_file, t + 1)

    for t in range(min(len(img_filenames), args.num_timesteps)):
        img_name = img_filenames[t]
        img = skio.imread(os.path.join(args.image_root, img_name))[:5, :5]
        enforce_spatial_constraint(img, output_file, t + 1)

    for t in range(min(len(img_filenames), args.num_timesteps)):
        query_label(img, output_file, t + 1)
    output_file.close()


if __name__ == '__main__':
    main()
