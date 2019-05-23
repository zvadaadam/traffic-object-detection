import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_img(dataset_path, image_path):

    print(os.path.join(dataset_path, image_path))

    return cv2.imread(os.path.join(dataset_path, image_path))


def add_bb_to_img(img, x_min, y_min, x_max, y_max, is_occulded=False):

    color = (255, 0, 0)  # blue is not occulded
    if is_occulded:
        color = (0, 0, 255)  # red is occulded

    pt1 = (x_min, y_min)
    pt2 = (x_max, y_max)

    return cv2.rectangle(img, pt1, pt2, color, 1)

def plot_img(img):

    plt.imshow(cv2.cvtColor(img, 4))
    plt.show()


def resize_image(img, new_width, new_height):
    return cv2.resize(img, (new_width, new_height, ), interpolation=cv2.INTER_CUBIC)

def evaluate_prediction(y_pred, y_true, iou_threshold=0.5, score_treshold=0.4):

    batch_size = y_true[0].shape[0]

def bbox_iou(A, B):

    intersect_mins = np.maximum(A[:, 0:2], B[:, 0:2])
    intersect_maxs = np.minimum(A[:, 2:4], B[:, 2:4])
    intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    A_area = np.prod(A[:, 2:4] - A[:, 0:2], axis=1)
    B_area = np.prod(B[:, 2:4] - B[:, 0:2], axis=1)

    iou = intersect_area / (A_area + B_area - intersect_area)

    return iou


def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])

    # Case in which they don't intersec --> max(,0)
    inter_area = max(xi2-xi1, 0)*max(yi2-yi1, 0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = float(inter_area)/float(union_area)

    return iou


def draw_boxes(image, boxes, labels):
    image_h, image_w, _ = image.shape

    for box in boxes:
        xmin = int(box.xmin * image_w)
        ymin = int(box.ymin * image_h)
        xmax = int(box.xmax * image_w)
        ymax = int(box.ymax * image_h)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        cv2.putText(image,
                    labels[box.get_label()] + ' ' + str(box.get_score()),
                    (xmin, ymin - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * image_h,
                    (0, 255, 0), 2)

    return image

# Discard all boxes with low scores and high IOU
# def non_max_suppression(boxes, scores, num_classes, max_boxes=50, score_thresh=0.3, iou_thresh=0.5):
#
#
#     boxes_list, label_list, score_list = [], [], []
#     max_boxes = tf.constant(max_boxes, dtype='int32')
#
#     # since we do nms for single image, then reshape it
#     boxes = tf.reshape(boxes, [-1, 4]) # '-1' means we don't konw the exact number of boxes
#     # confs = tf.reshape(confs, [-1,1])
#     score = tf.reshape(scores, [-1, num_classes])
#
#     # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
#     mask = tf.greater_equal(score, tf.constant(score_thresh))
#     # Step 2: Do non_max_suppression for each class
#     for i in range(num_classes):
#         # Step 3: Apply the mask to scores, boxes and pick them out
#         filter_boxes = tf.boolean_mask(boxes, mask[:, i])
#         filter_score = tf.boolean_mask(score[:, i], mask[:, i])
#         nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
#                                                    scores=filter_score,
#                                                    max_output_size=max_boxes,
#                                                    iou_threshold=iou_thresh, name='nms_indices')
#         label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
#         boxes_list.append(tf.gather(filter_boxes, nms_indices))
#         score_list.append(tf.gather(filter_score, nms_indices))
#
#     boxes = tf.concat(boxes_list, axis=0)
#     score = tf.concat(score_list, axis=0)
#     label = tf.concat(label_list, axis=0)
#
#     return boxes, score, label
