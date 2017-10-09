import os
import glob
import json
import argparse
import pickle
import collections
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d
from util import AP, prcurve


def main(args):

    gt_path = os.path.join(args.root_dir, 'detection', 'gt.txt')
    target_files = glob.glob(os.path.join(args.root_dir, 'detection', '*.cache'))

    with open(gt_path, 'r') as f:
        lines = f.readlines()
        gt = []
        for line in lines:
            name, x1, y1, x2, y2, cl = line.split()
            x1 = max(0, float(x1)); y1 = max(0, float(y1))
            x2 = min(1, float(x2)); y2 = min(1, float(y2))
            gt.append({'name': name,
                       'bbox': [x1, y1, x2, y2],
                       'class': cl,
                       'detected': False})

    class_instances = [g['class'] for g in gt]
    class_counter = collections.Counter(class_instances)
    class_keys = list(class_counter)

    # main graph
    gs = gridspec.GridSpec(2, len(class_keys))
    ax = plt.subplot(gs[0, :])
    names = []
    for target_file in target_files:
        name, recall, precision = prcurve(target_file, threshold=args.threshold)
        ax.plot(recall, precision)
        names.append(name)
    ax.axis([0, 1, 0, 1])
    plt.legend(names)

    for i, key in enumerate(class_keys):
        ax = plt.subplot(gs[1, i])
        for target_file in target_files:
            name, recall, precision = prcurve(target_file, category=key, threshold=args.threshold)
            ax.plot(recall, precision)
        ax.axis([0, 1, 0, 1])
        ax.set_title(key)
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hello, deep learning world!')
    parser.add_argument("--root_dir", default='/media/phantom/World/phantom_benchmark', help="benchmark root directory")
    parser.add_argument("--threshold", default=0.5, help="IOU threshold")
    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        print("Couldn't find root directory: %s" % args.root_dir)

    main(args)

