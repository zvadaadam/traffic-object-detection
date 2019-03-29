import os
import cv2
import matplotlib.pyplot as plt


def load_img(dataset_path, image_path):
    return cv2.imread(os.path.join(dataset_path, image_path))


def add_bb_to_img(img, x_min, y_min, x_max, y_max, is_occulded=False):

    color = (255, 0, 0)  # blue is not occulded
    if is_occulded:
        color = (0, 0, 255)  # red is occulded

    return cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 3)

def plot_img(img):

    plt.imshow(cv2.cvtColor(img, 4))
    plt.show()


def resize_image(img, new_height, new_width):
    return cv2.resize(img, (new_height, new_width), interpolation=cv2.INTER_CUBIC)
