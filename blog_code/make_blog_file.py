import argparse
import numpy as np
import os
import shutil
import skimage.io as skio

parser = argparse.ArgumentParser(
    description="Read frames from input file and generate BLOG file"
                "for background subtraction.") 
parser.add_argument('--crop', type=int,
    help="Crop the image from the top-left corner to args.crop-by-args.crop")
parser.add_argument('--init_mean',
    help="Path to images for initialization of Particle Filter means")
parser.add_argument('--image_root',
    help="Directory where images to process are located")
parser.add_argument('--input_name', default='bsub.dblog',
    help="Name of input template file (default: bsub.dblog)")
parser.add_argument('--num_timesteps', type=int, default=10,
    help="Number of timesteps to read (default: 10)")
parser.add_argument('--output_name', default='output_blog.dblog',
    help="Name of output BLOG file (default: output_blog.dblog)")
parser.add_argument('--offline', action='store_true',
    help="Boolean flag to specify offline models")
parser.add_argument('--query_type', default='label',
    help="Type of query (label, mean, etc)")
parser.add_argument('--start_time', type=int, default=0,
    help="Start of image sequence (default: 0)")
parser.add_argument('--swift', action='store_true',
    help="Boolean flag to specify swift conventions")
parser.set_defaults(swift=False)
args = parser.parse_args()

def read_img_intensity(img, output_file, t):
    height, width = img.shape[:2]
    if args.offline:
        obs_template = 'obs Intensity(ImageX[%d], ImageY[%d], Time[%d]) = ' \
                       '[%0.1f; %0.1f; %0.1f];\n'
    else:
        obs_template = 'obs Intensity(ImageX[%d], ImageY[%d], @%d) = ' \
                       '[%0.1f; %0.1f; %0.1f];\n'

    for i in range(height):
        for j in range(width):
            r, g, b = img[i, j].tolist()
            obs = obs_template % (i, j, t, r, g, b)
            output_file.write(obs)

def init_mean_component(img, comp, output_file):
    height, width = img.shape[:2]
    mean_template = 'obs Mean(Component[%d], ImageX[%d], ImageY[%d], @%d) = ' \
                    '[%0.1f; %0.1f; %0.1f];\n'
    for i in range(height):
        for j in range(width):
            r, g, b = img[i, j].tolist()
            mean = mean_template % (comp, i, j, 1, r, g, b)
            output_file.write(mean)

def enforce_spatial_constraint(img, output_file, t):
    height, width = img.shape[:2]
    if args.offline:
        obs_template = 'obs Output(ImageX[{0}], ImageY[{1}], ' \
                       'ImageX[{2}], ImageY[{3}], Time[{4}]) = true;\n'
    else:
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
    if args.query_type == 'label':
        if args.offline:
            label_template = 'query Label(ImageX[{0}], ImageY[{1}], Time[{2}]);\n'
        else:
            label_template = 'query Label(ImageX[{0}], ImageY[{1}], @{2});\n'

        for t in range(1, t + 1):
            for i in range(height):
                for j in range(width):
                    label = label_template.format(i, j, t)
                    output_file.write(label)
    elif args.query_type == 'mean':
        if args.offline:
            label_template = 'query Mean(Component[{0}], ImageX[{1}], ' \
                                        'ImageY[{2}]);\n' \
                             'query Variance(Component[{0}], ImageX[{1}], ' \
                                            'ImageY[{2}]);\n'
        else:
            label_template = 'query Mean(Component[{0}], ImageX[{1}], ' \
                                        'ImageY[{2}], @{3});\n' \
                             'query Variance(Component[{0}], ImageX[{1}], ' \
                                            'ImageY[{2}], @{3});\n'

        for c in range(3):
            for i in range(height):
                for j in range(width):
                    if args.offline:
                        label = label_template.format(c, i, j)
                    else:
                        label = label_template.format(c, i, j, t)
                    output_file.write(label)
    elif args.query_type == 'associated_component':
        if args.offline:
            label_template = 'query AssociatedComponent(ImageX[{0}], ' \
                                          'ImageY[{1}], Time[{2}]);\n'
        else:
            label_template = 'query AssociatedComponent(ImageX[{0}], ' \
                                                        'ImageY[{1}], @{2});\n'
        for t in range(1, t + 1):
            for i in range(height):
                for j in range(width):
                    label = label_template.format(i, j, t)
                    output_file.write(label)


def add_prev_offline(output_file, t):
    output_file.write('fixed Time prevTime(Time t) =\n')
    output_file.write('    case t in {\n')
    output_file.write('        Time[0] -> Time[0],\n')
    for i in range(1, t):
        output_file.write('        Time[%d] -> Time[%d],\n' % (i, i - 1))
    output_file.write('        Time[%d] -> Time[%d]\n' % (t, t - 1))
    output_file.write('    };\n\n')


def main():
    shutil.copyfile(args.input_name, args.output_name)
    output_file = open(args.output_name, 'a')
    img_filenames = os.listdir(args.image_root)

    if args.offline:
        add_prev_offline(output_file, args.num_timesteps)

    for t in range(min(len(img_filenames), args.num_timesteps)):
        img_name = img_filenames[t]
        img = skio.imread(os.path.join(args.image_root, img_name))
        if args.crop:
            img = img[:args.crop, :args.crop]
        read_img_intensity(img, output_file, t + 1)

    if args.init_mean:
        for i in range(1, 4):
            img = skio.imread(os.path.join(args.init_mean, 'img%d.png' % i))
            if args.crop:
                img = img[:args.crop, :args.crop]
            init_mean_component(img, i - 1, output_file)

    for t in range(min(len(img_filenames), args.num_timesteps)):
        img_name = img_filenames[t]
        img = skio.imread(os.path.join(args.image_root, img_name))
        if args.crop:
            img = img[:args.crop, :args.crop]
        enforce_spatial_constraint(img, output_file, t + 1)

    query_label(img, output_file, t + 1)
    output_file.close()


if __name__ == '__main__':
    main()
