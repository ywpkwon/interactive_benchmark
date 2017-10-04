import os
import glob
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict


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
        if p['name'] not in gt: continue
        cadidates = gt[p['name']]
        for g in cadidates:
            if g['detected']: continue

            iou_test = iou(boxxy2boxwh(g['bbox']), boxxy2boxwh(p['bbox'])) > iou_threshold
            class_test = True  #g['class'].lower() == p['class'].lower()
            already_test = not g['detected']

            if iou_test and class_test and already_test:
                g['detected'] = True
                p['correct'] = True
                continue

    true_positives = np.array([p['correct'] for p in pr], dtype=np.int32)
    true_positives = np.cumsum(true_positives)
    recall = true_positives / n_gt_bboxes
    precision = true_positives / (np.arange(len(true_positives))+1)

    print (outname, AP(recall, precision))
    result = {"gt": gt, "prediction": pr}

    return outname, result

def main(cfg):

    gt_path = 'gt.txt'
    # pred_path = '/home/phantom/Documents/benchmark/ssd_incep1_wide480.out'
    # evaluate(gt_path, pred_path)

    # pred_path = '/home/phantom/Documents/benchmark/ssd_wide480a.out'
    # evaluate(gt_path, pred_path)
    files2evaluate = glob.glob("*.out")
    files2evaluate = [f for f in files2evaluate if os.path.basename(f)[0] != '_']
    print (files2evaluate)

    for file in files2evaluate:
        outname, result = evaluate(gt_path, file, 0.5)
        with open(outname + '.pickle', 'wb') as pf:
            pickle.dump(result, pf)


    # pred_path = "ph_ssd_wide480a.txt"
    # pred_path = "ph_ssd_incep1_wide480.txt"
    # evaluate(gt_path, pred_path, 0.5)
    # pred_path = "ph_ssd_incep_wide480_notw.txt"
    # evaluate(gt_path, pred_path, 0.5)
    # pred_path = "ph_ssd_incep_wide480_notw2.txt"
    # evaluate(gt_path, pred_path, 0.5)


    # if cfg.show:
    #     drawer = Drawer(cfg.common.classes)
    #     plt.grid(False)
    #     plt.ion()

    # if cfg.monitor:
    #     ckpt_dir = os.path.join(cfg.path.ckpt_dir, cfg.cfg_name)
    #     ckpt_provider = CkptProvider(ckpt_dir=ckpt_dir)
    #     cfg.show = False
    # else:
    #     ckpt_provider = CkptProvider(ckpt_list=cfg.eval_weights)
    #     if ckpt_provider.size()==0:
    #         print('-- There is no weight files.')
    #         sys.exit(0)

    # setname = cfg.eval_set
    # dataset = dataset_factory.get_dataset(cfg['dataset'], setname, '../datasets/cache')
    # num_examples = dataset.num_samples
    # print('Dataset:', dataset.data_sources, '|', dataset.num_samples)

    # name, image, labels, bboxes, original_shape = _get_example(cfg, dataset, is_training=False)

    # net = YOLO(cfg, None, is_training=False)

    # saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

    # # GPU configuration
    # if cfg.num_gpus == 0:   config = tf.ConfigProto(device_count={'GPU': 0})
    # else:                   config = tf.ConfigProto()

    # with tf.Session(config=config) as sess:

    #     print ('initializing queue ...')
    #     with slim.queues.QueueRunners(sess):

    #         while True:
    #             weight_file = ckpt_provider.pop()
    #             if not weight_file:
    #                 if cfg.monitor: time.sleep(60); continue
    #                 else: break

    #             sess.run(tf.local_variables_initializer())
    #             sess.run(tf.global_variables_initializer())

    #             outputf = "{}_{}_{}.out.pickle".format(cfg['dataset'], setname, os.path.splitext(os.path.basename(weight_file))[0])
    #             outputf = os.path.join(os.path.dirname(weight_file), outputf)
    #             if os.path.isfile(outputf): continue

    #             print ('restoring (%s) ...' % weight_file)
    #             saver.restore(sess, weight_file)

    #             results = []
    #             n_real_bboxes = 0

    #             # instead of batches, we go through each example.
    #             for _ in tqdm(range(num_examples)):

    #                 gname, gimage, glabels, gbboxes, oshape = sess.run([name, image, labels, bboxes, original_shape])
    #                 gname = gname.decode("ascii")
    #                 gbimage = np.expand_dims(gimage, 0)

    #                 feed_dict = {net.x: gbimage}
    #                 [final_scores, final_bboxes, final_classes] = sess.run([
    #                                                 net.layers['selected_scores'],
    #                                                 net.layers['selected_boxes'],
    #                                                 net.layers['selected_classes']],
    #                                                 feed_dict=feed_dict)

    #                 final_scores = final_scores[0, :]
    #                 final_bboxes = final_bboxes[0, :]
    #                 final_classes = final_classes[0, :]

    #                 # xywh -> yxyx
    #                 final_bboxes = np.concatenate([final_bboxes[:,[1]]-final_bboxes[:,[3]]*0.5,
    #                                         final_bboxes[:,[0]]-final_bboxes[:,[2]]*0.5,
    #                                         final_bboxes[:,[1]]+final_bboxes[:,[3]]*0.5,
    #                                         final_bboxes[:,[0]]+final_bboxes[:,[2]]*0.5], axis=1)

    #                 final_class_id = np.argmax(final_classes, axis=1)
    #                 # final_class_prob = np.max(final_classes, axis=1)
    #                 # final_scores *= final_class_prob

    #                 result = [{'name':gname, 'bbox':final_bboxes[i], 'class':final_class_id[i], 'score':final_scores[i], 'correct':False} for i in range(len(final_scores))]
    #                 truth = [{'bbox':gbboxes[i], 'class':glabels[i], 'detected':False} for i in range(len(glabels))]
    #                 result.sort(key=lambda x: -x['score'])     # sort results by descending score

    #                 for titem in truth:
    #                     # if titem is valid
    #                     n_real_bboxes += 1

    #                     for ritem in result:
    #                         if iou(boxxy2boxwh(titem['bbox']), boxxy2boxwh(ritem['bbox']))>0.5 and titem['class'] == ritem['class']:
    #                             ritem['correct']=True

    #                 results += result

    #                 if cfg.show:
    #                     # image_ = cv2.cvtColor((gimage+1)*128, cv2.COLOR_RGB2BGR).astype(np.uint8)
    #                     image_ = ((gimage+1)*128).astype(np.uint8)
    #                     bboxes_ = np.concatenate([
    #                         (final_bboxes[:,[1]]+final_bboxes[:,[3]])*0.5,
    #                         (final_bboxes[:,[0]]+final_bboxes[:,[2]])*0.5,
    #                         final_bboxes[:,[3]]-final_bboxes[:,[1]],
    #                         final_bboxes[:,[2]]-final_bboxes[:,[0]]], 1)
    #                     bboxes_ *= [cfg.common.image_width, cfg.common.image_height, cfg.common.image_width, cfg.common.image_height]
    #                     valid = final_scores>0.2
    #                     classes_id_ = final_class_id[valid]
    #                     scores_ = final_scores[valid]
    #                     bboxes_ = bboxes_[valid,:]
    #                     objects = [[cfg.common.classes[classes_id_[i]]] + bboxes_[i].tolist() + [scores_[i]] for i in range(scores_.shape[0])]
    #                     drawer = Drawer(cfg.common.classes, show_label=False)
    #                     imgout = drawer.draw_result(image_, objects)
    #                     plt.imshow(imgout)
    #                     plt.waitforbuttonpress()

    #             # sort results by descending score
    #             results.sort(key=lambda x: -x['score'])

    #             save = {}
    #             save['results'] = results
    #             save['n_real_bboxes'] = n_real_bboxes

    #             with open(outputf, 'wb') as of:
    #                 pickle.dump(save, of)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hello, deep learning world!')
    # parser.add_argument("--conf", required=True, action="store", help="training configuration file")
    parser.add_argument("--gpuid", default="0", action="store", help="specify which gpu to turn on. e.g., 0,1,2")
    parser.add_argument("--eval_set", default='val', action="store", help="specify split name to evaluate")
    parser.add_argument("--show", action='store_true', help="just display result and not save.")
    parser.add_argument("--monitor", action='store_true', help="evaluate every checkpoint files while waiting.")
    parser.add_argument("--threshold", default=0.3, type=float, help="confidence threshold")
    parser.add_argument("--iou_threshold", default=0.3, type=float, help="IoU threshold")

    args = parser.parse_args()
    # for evaluation (PR curve), one should consider (almost) all output.
    # cfg['test']['threshold'] = 0.01
    main(args)