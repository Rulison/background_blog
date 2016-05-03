import argparse
import os
import sys

parser = argparse.ArgumentParser(
    description="Given input prediction and ground truth files,"
                "evaluate prediction accuracy.")
parser.add_argument('-p', '--prediction',
                    help="Path to prediction image file")
parser.add_argument('-t', '--truth',
                    help="Path to ground truth image file")
args = parser.parse_args()
