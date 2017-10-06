import os
import glob
import json
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from util import iou, boxxy2boxwh, AP
import matplotlib.pyplot as plt


def evaluate(gt_path, pred_path, iou_threshold):

    outname = os.path.splitext(os.path.basename(pred_path))[0]

    with open(gt_path, 'r') as f:
        lines = f.readlines()
        gt = {}
        n_gt_bboxes = 0
        for line in lines:
            name, x1, y1, x2, y2, cl = line.split()
            x1 = max(0, float(x1)); y1 = max(0, float(y1))
            x2 = min(1, float(x2)); y2 = min(1, float(y2))
            if name not in gt: gt[name] = []
            gt[name].append({'name': name,
                             'bbox': [x1, y1, x2, y2],
                             'class': cl.lower(),
                             'detected': False,
                             'valid': True})
            n_gt_bboxes += 1   # if valid, e.g., difficulty

    with open(pred_path, 'r') as f:
        lines = f.readlines()
        pr = []
        for line in lines:
            name, prob, x1, y1, x2, y2, cl = line.split()
            # name, x1, y1, x2, y2, prob,  cl = line.split()
            x1, x2 = np.sort([x1, x2])
            y1, y2 = np.sort([y1, y2])
            x1 = max(0, float(x1)); y1 = max(0, float(y1))
            x2 = min(1, float(x2)); y2 = min(1, float(y2))
            pr.append({'name': name,
                       'prob': float(prob),
                       'bbox': [x1, y1, x2, y2],
                       'class': cl.lower(),
                       'correct': False})

    pr.sort(key=lambda x: -x['prob'])

    # gt_fileset = set([g['name'] for g in gt])
    # pr_fileset = set([p['name'] for p in pr])

    # common_fileset = [v for v in pr_fileset if v in gt_fileset]
    # if len(common_fileset) != len(pr_fileset):
        # print("ERROR! prediction set contains images that are not in the ground truth.")
        # print([v for v in pr_fileset if v not in gt_fileset])
        # quit()

    for p in tqdm(pr):
        if p['name'] not in gt:
            """
            if filename is not GT, it means that there was not object in that file
            """
            continue
        cadidates = gt[p['name']]
        for g in cadidates:
            if g['detected']: continue
            iou_test = iou(boxxy2boxwh(g['bbox']), boxxy2boxwh(p['bbox'])) > iou_threshold

            # import pdb; pdb.set_trace()
            # plt.imshow('/media/phantom/World/phantom_benchmark/images/' + p['name'] + '.png')
            # plt.plot([p['bbox'][0] * 1280, p['bbox'][2] * 1280], [p['bbox'][1] * 720, p['bbox'][3] * 720])
            # plt.show()

            class_test = True # g['class'].lower() == p['class'].lower()
            already_test = not g['detected']

            if iou_test and class_test and already_test:
                g['detected'] = True
                p['correct'] = True
                break


    true_positives = np.array([p['correct'] for p in pr], dtype=np.float32)
    true_positives = np.cumsum(true_positives)
    recall = true_positives / n_gt_bboxes
    precision = true_positives / (np.arange(len(true_positives))+1)
    ap = AP(recall, precision)
    result = {"gt": gt, "prediction": pr}
    return ap, result, recall, precision

def main(cfg, plot=False):

    with open('setting.json') as jf:
        cfg = json.load(jf)

    benchmark_root = cfg['benchmark_root']
    gt_path = os.path.join(benchmark_root, cfg['target_dir'], 'gt.txt')
    det_dir = os.path.join(benchmark_root, cfg['target_dir'])

    files2evaluate = glob.glob(os.path.join(det_dir, "*.out"))
    files2evaluate = [f for f in files2evaluate if os.path.basename(f)[0] != '_']
    print (files2evaluate)

    alg_names = []
    for file in files2evaluate:
        name = os.path.splitext(os.path.basename(file))[0]
        outpath = os.path.join(det_dir, name + '.cache')
        if os.path.isfile(outpath): continue

        ap, result, recall, precision = evaluate(gt_path, file, 0.5)
        print ("{}: AP {}".format(name, ap))

        with open(outpath, 'wb') as pf:
            pickle.dump(result, pf)

        if plot:
            plt.plot(recall, precision)
            alg_names.append(name)

    if len(alg_names)>0 and plot:
        plt.legend(alg_names)
        plt.axis([0, 1, 0, 1])
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hello, deep learning world!')
    args = parser.parse_args()
    main(args, plot=True)