import numpy as np


def AP(rec, prec):
    """
    There are several types of average precision (AP).
    Here follows the VOC style. As to my knowledge,
    it may be called interpolated AP, and values can be slightly higher than usual AP.
    """
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    t = np.nonzero(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[t]-mrec[t-1])*mpre[t])
    return ap


def boxxy2boxwh(boxxy):
  """Convert box representation
  [x1, y1, x2, y2] --> [cx, cy, width, height],
  Args:
    boxxy: array of 4 elements [x1, y1, x2, y2].
  Returns:
    boxwh: array of 4 elements [cx, cy, width, height]
  """
  boxwh = [0]*4
  boxwh[0] = 0.5*(boxxy[0]+boxxy[2])
  boxwh[1] = 0.5*(boxxy[1]+boxxy[3])
  boxwh[2] = boxxy[2]-boxxy[0]
  boxwh[3] = boxxy[3]-boxxy[1]
  assert boxwh[2]>=0 and boxwh[3]>=0
  return boxwh


def iou(box1, box2):
  """Compute the Intersection-Over-Union of two given boxes.
     Note that boxes are [cx, cy, width, height],
                **NOT** [x1, y1, x2, y2]!!
  Args:
    box1: array of 4 elements [cx, cy, width, height].
    box2: same as above
  Returns:
    iou: a float number in range [0, 1]. iou of the two boxes.
  """

  lr = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
      max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
  if lr > 0:
    tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
        max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
    if tb > 0:
      intersection = tb*lr
      union = box1[2]*box1[3]+box2[2]*box2[3]-intersection

      return intersection/union

  return 0


def batch_iou(boxes, box):
  """Compute the Intersection-Over-Union of a batch of boxes with another
  box.

  Args:
    box1: 2D array of [cx, cy, width, height].
    box2: a single array of [cx, cy, width, height]
  Returns:
    ious: array of a float number in range [0, 1].
  """
  lr = np.maximum(
      np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
      np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
      0
  )
  tb = np.maximum(
      np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
      np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
      0
  )
  inter = lr*tb
  union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
  return inter/union
