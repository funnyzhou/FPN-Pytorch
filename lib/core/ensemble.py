import sys
sys.path.append('E:\mask-rcnn.pytorch\lib')
import utils.boxes as box_utils
import json
import os
from copy import deepcopy
import numpy as np

def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_keyps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes, all_segms, all_keyps

def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for cls_idx in range(1, len(im_res)):
        all_res[cls_idx][index] = im_res[cls_idx]

def GeneralEnsemble(dets, iou_thresh=0.5, weights=None):
    assert (type(iou_thresh) == float)

    ndets = len(dets)

    if weights is None:
        w = 1 / float(ndets)
        weights = [w] * ndets
    else:
        assert (len(weights) == ndets)

        s = sum(weights)
        for i in range(0, len(weights)):
            weights[i] /= s

    out = list()
    used = list()

    for idet in range(0, ndets):
        det = dets[idet]
        for box in det:
            if box in used:
                continue

            used.append(box)
            # Search the other detectors for overlapping box of same class
            found = []
            for iodet in range(0, ndets):
                odet = dets[iodet]

                if odet == det:
                    continue

                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    if not obox in used:
                        # Not already used
                        if box[4] == obox[4]:
                            # Same class
                            iou = computeIOU(box, obox)
                            if iou > bestiou:
                                bestiou = iou
                                bestbox = obox

                if not bestbox is None:
                    w = weights[iodet]
                    found.append((bestbox, w))
                    used.append(bestbox)

            # Now we've gone through all other detectors
            if len(found) == 0:
                new_box = list(box)
                new_box[5] /= ndets
                out.append(new_box)
            else:
                allboxes = [(box, weights[idet])]
                allboxes.extend(found)

                xc = 0.0
                yc = 0.0
                bw = 0.0
                bh = 0.0
                conf = 0.0

                wsum = 0.0
                for bb in allboxes:
                    w = bb[1]
                    wsum += w

                    b = bb[0]
                    xc += w * b[0]
                    yc += w * b[1]
                    bw += w * b[2]
                    bh += w * b[3]
                    conf += w * b[5]

                xc /= wsum
                yc /= wsum
                bw /= wsum
                bh /= wsum

                new_box = [xc, yc, bw, bh, box[4], conf]
                out.append(new_box)
    return out


def getCoords(box):
    x1 = float(box[0])
    x2 = float(box[0]) + float(box[2])
    y1 = float(box[1])
    y2 = float(box[1]) + float(box[3])
    return x1, x2, y1, y2


def computeIOU(box1, box2):
    x11, x12, y11, y12 = getCoords(box1)
    x21, x22, y21, y22 = getCoords(box2)

    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou


# if __name__ == "__main__":
data_dir = '../ensemble/'

file_names = os.listdir(data_dir)
bboxes_dict = {}
scores_dict = {}
all_bboxes = {}
all_scores = {}
all_ids = {}

# method = "Weighted Union"
method = 'SOFT_NMS'

ensemble_results = []
if method == "Weighted Union":
    for n in file_names:
        f = open(data_dir + n, 'r')
        bboxes = json.load(f)
        for i in range(len(bboxes)):
            if not str(bboxes[i]['image_id']) in bboxes_dict.keys() and bboxes[i]['bbox'][2] < 800 and bboxes[i]['bbox'][3] < 800:
                bboxes_dict[str(bboxes[i]['image_id'])] = []
                scores_dict[str(bboxes[i]['image_id'])] = []
            if bboxes[i]['bbox'][2] < 800 and bboxes[i]['bbox'][3] < 800:
                bboxes[i]['bbox'].extend([1, bboxes[i]['score']])
                bboxes_dict[str(bboxes[i]['image_id'])].append(bboxes[i]['bbox'])

            if not str(bboxes[i]['image_id']) in all_ids.keys():
                all_ids[str(bboxes[i]['image_id'])] = []

        all_bboxes[n] = deepcopy(bboxes_dict)
        bboxes_dict = {}
        scores_dict = {}
        print(n)

    for k in all_ids.keys():
        bboxes = []
        for n in all_bboxes.keys():
            if not k in all_bboxes[n].keys():
                bboxes.append([])
            else:
                bboxes.append(all_bboxes[n][k])
        if k == '143':
            print('haha')
        ens = GeneralEnsemble(bboxes, weights=[1, 0.6, 0.15, 0.2, 0.1, 0.15, 0.15, 0.2, 0.15, 0.15, 0.15])
        try:
            if len(ens):
                ens = np.array(ens)
                if len(ens.shape) == 1:
                    ids = np.argsort(-ens)
                else:
                    ids = np.argsort(-ens[:, -1])
                ens = ens[ids].tolist()
        except Exception as e:
            print('haha')
            print(ens)
        for id in range(len(ens)):
            ensemble_results.append(
                {'image_id': k, 'category_id': 1, 'bbox': ens[id][0:4], 'score': ens[id][5]})

        print(k)

    f2 = open('../results/ensemble_results.json', 'w')
    json.dump(ensemble_results, f2)
    print('finished')

elif method == 'SOFT_NMS':
    all_boxes, all_segms, all_keyps = empty_results(2, 1000)
    weights = [1, 0.6, 0.15, 0.2, 0.1, 0.15, 0.15, 0.2, 0.15, 0.15, 0.15]
    c = 0
    for n in file_names:
        f = open(data_dir + n, 'r')
        bboxes = json.load(f)
        for i in range(len(bboxes)):
            if not str(bboxes[i]['image_id']) in bboxes_dict.keys() and bboxes[i]['bbox'][2] < 800 and bboxes[i]['bbox'][3] < 800:
                bboxes_dict[str(bboxes[i]['image_id'])] = []
                scores_dict[str(bboxes[i]['image_id'])] = []
            if bboxes[i]['bbox'][2] < 800 and bboxes[i]['bbox'][3] < 800:
                bboxes_dict[str(bboxes[i]['image_id'])].append(bboxes[i]['bbox'])
                scores_dict[str(bboxes[i]['image_id'])].append(bboxes[i]['score'] * weights[c])

            if not str(bboxes[i]['image_id']) in all_ids.keys():
                all_ids[str(bboxes[i]['image_id'])] = []
        print(n)
        c += 1

    c = 0
    for k in bboxes_dict.keys():
        bboxes = np.array(bboxes_dict[k])
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        scores = np.array(scores_dict[k])
        scores = np.reshape(scores, (scores.shape[0], 1))
        dets_j = np.hstack((bboxes, scores)).astype(np.float32, copy=False)

        nms_dets, _ = box_utils.soft_nms(
            dets_j,
            sigma=0.3,
            overlap_thresh=0.5,
            score_thresh=0.0001,
            method='gaussian'
        )
        # keep = box_utils.nms(dets_j, 0.5)
        # nms_dets = dets_j[keep, :]

        for j in range(nms_dets.shape[0]):
            ensemble_results.append({'image_id': k, 'category_id': 1, 'bbox': nms_dets[j][0:4].tolist(), 'score': nms_dets[j][4].tolist()})
        print(k)

    f2 = open('../results/ensemble_results.json', 'w')
    json.dump(ensemble_results, f2)
    print('finished')

    # det_file = os.path.join(output_dir, det_name)
    # save_object(
    #     dict(
    #         all_boxes=all_boxes,
    #         all_segms=all_segms,
    #         all_keyps=all_keyps,
    #         cfg=cfg_yaml
    #     ), det_file
    # )

# USE WEIGHTED UNION
# ens = GeneralEnsemble(dets, weights=[1.0, 0.1, 0.5])
# USE NMS
