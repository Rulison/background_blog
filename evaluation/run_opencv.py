import argparse
import cv2
import numpy as np
import os
import skimage.io as skio
import sys

parser = argparse.ArgumentParser(
    description="Evaluate image sequence using OpenCV benchmarks")
parser.add_argument('-v', '--version', type=int,
                    help="Which benchmark to evaluate [1, 2, 3]")
parser.add_argument('-f', '--filepath',
                    help="Path to directory containing image sequence")
parser.add_argument('-o', '--output',
                    help="Path to output directory for predictions",
                    default="opencv_output")
args = parser.parse_args()


def apply_background_subtractor(fgbg, img_filenames, imgpath_prefix):
    masks = []
    for img in img_filenames: 
        curr_img = skio.imread(imgpath_prefix + img)
        mask = fgbg.apply(curr_img)
        masks.append(mask)
    return masks

def save_images(file_string, images):
    i = 0
    for img in images:
        skio.imsave(file_string.format(i), img)
        i += 1

def main():
    fgbg = None
    if args.version == 1:
        fgbg = cv2.BackgroundSubtractorMOG()
    elif args.version == 2:
        fgbg = cv2.BackgroundSubtractorMOG2()
    elif args.version == 3:
        fgbg = cv2.BackgroundSubtractorGMG()
    else:
        print "Please enter a version number between 1 and 3"
        return

    # Create output directory if does not exist
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # Read images from specified directory
    filenames = os.listdir(args.filepath)
    filenames = [x for x in filenames if x[0] != '.']
    masks = apply_background_subtractor(fgbg, filenames, args.filepath) 
    save_images(args.output + '/output_im_{}.png', masks)

if __name__ == '__main__':
    main()
