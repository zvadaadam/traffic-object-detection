from object_detection.utils import image_utils


def to_yolo_cords(x1, y1, x2, y2, img_shape, relative_cords=False):

    image_height, image_width, _ = img_shape

    def sorting(l1, l2):
        if l1 > l2:
            lmax, lmin = l1, l2
            return lmax, lmin
        else:
            lmax, lmin = l2, l1
            return lmax, lmin

    xmax, xmin = sorting(x1, x2)
    ymax, ymin = sorting(y1, y2)

    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0

    w = xmax - xmin
    h = ymax - ymin

    if relative_cords:
        dw = 1. / image_width
        dh = 1. / image_height

        x = x * dw
        w = w * dw

        y = y * dh
        h = h * dh

    return x, y, w, h


def from_yolo_to_cord(box, shape):
    """
    Method converts YOLO box cords (x, y, w, h) to corner box cords (x_min, y_min, x_max, y_max)
    :param box: (x, y, w, h)
    :param shape: image shape (w, h, 3)
    :return: (x_min, y_min, x_max, y_max)
    """

    image_height, image_width, _ = shape
    x, y, w, h = box[0], box[1], box[2], box[3]

    # x1, y1 = ((x + witdth)/2)*img_width, ((y + height)/2)*img_height
    #x1, y1 = int((box[0] + box[2] / 2) * img_w), int((box[1] + box[3] / 2) * img_h)

    # x2, y2 = ((x - witdth)/2)*img_width, ((y - height)/2)*img_height
    #2, y2 = int((box[0] - box[2] / 2) * img_w), int((box[1] - box[3] / 2) * img_h)

    x_min = int((x - w/2) * image_width)
    y_min = int((y - h/2) * image_height)

    x_max = int((x + w/2) * image_width)
    y_max = int((y + h/2) * image_height)

    return x_min, y_min, x_max, y_max



def img_to_yolo_shape(img):

    # TODO: move
    yolo_image_height = 448
    yolo_image_width = 448

    img = image_utils.resize_image(img, yolo_image_height, yolo_image_width)

    return img


def resize_cords(cords, image_shape, new_image_shape, relative_cords=False):
    """
    Resize the coordinate to new image shape
    :param cords: [(x, y, w, h)]
    :param image_shape: (x, y, z)
    :param new_image_shape: (x, y, z)
    :param relative_cords: Flag for converting the coordinate to interval <0,1>
    :return: [(x', y', w', h')]
    """
    # TODO: move
    new_image_height, new_image_width, _ = new_image_shape
    image_height, image_width, _ = image_shape

    ratio_height = new_image_height / image_height
    ratio_width = new_image_width / image_width

    if relative_cords:
        dh = 1. / new_image_height
        dw = 1. / new_image_width
    else:
        dh = 1
        dw = 1

    new_cords = []
    for cord in cords:
        x = cord[0] * ratio_width * dw
        y = cord[1] * ratio_height * dh
        w = cord[2] * ratio_width * dw
        h = cord[3] * ratio_height * dh

        new_cords.append((x, y, w, h))

    return new_cords


def grid_index(x, y):

    # TODO: read from config
    yolo_image_width = 448
    yolo_image_height = 448

    grid_size = 7

    return int(x/(1/grid_size )), int(y/(1/grid_size))

