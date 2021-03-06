import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image


def load_img(dataset_path, image_path):
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


def draw_boxes_cv(image, boxes, scores, classes):
    image_h, image_w, _ = image.shape

    for box, score, object_class in zip(boxes, scores, classes):

        label = '{} {:.2f}'.format(object_class, score)

        x_min, y_min, x_max, y_max = box

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_h, x_max)
        y_max = min(image_w, y_max)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv2.putText(image, label, (int(x_min), int(y_min - 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.25, (0, 255, 0), 1, cv2.LINE_AA)

    return image


def draw_boxes_PIL(image, boxes, scores, classes):

    if np.max(image) <= 1.0:
        image = image * 255

    #cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(np.uint8(image))

    #font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=2)
    font = ImageFont.load_default()
    #thickness = (image.size[0] + image.size[1]) // 300
    thickness = 1

    for box, score, object_class in zip(boxes, scores, classes):

        label = '{} {:.2f}'.format(object_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        x_min, y_min, x_max, y_max = box

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.size[0], x_max)
        y_max = min(image.size[1], y_max)

        print(label, (x_min, y_min), (x_max, y_max))

        if y_min - label_size[1] >= 0:
            text_origin = np.array([x_min, y_min - label_size[1]])
        else:
            text_origin = np.array([x_min, y_min + 1])

        for i in range(thickness):
            draw.rectangle([x_min + i, y_min + i, x_max - i, y_max - i])

        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(255, 0, 0))
        draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)

        del draw

    return np.array(image.convert('RGB'))


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
