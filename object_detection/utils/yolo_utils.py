from object_detection.utils import image_utils

def from_cord_to_yolo(x1, y1, x2, y2, img_shape):

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

    dw = 1. / image_width
    dh = 1. / image_height

    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0

    w = xmax - xmin
    h = ymax - ymin

    x = x * dw
    w = w * dw

    y = y * dh
    h = h * dh

    return (x, y, w, h)


def from_yolo_to_cord(box, shape):

    img_h, img_w, _ = shape

    # x1, y1 = ((x + witdth)/2)*img_width, ((y + height)/2)*img_height
    x1, y1 = int((box[0] + box[2] / 2) * img_w), int((box[1] + box[3] / 2) * img_h)

    # x2, y2 = ((x - witdth)/2)*img_width, ((y - height)/2)*img_height
    x2, y2 = int((box[0] - box[2] / 2) * img_w), int((box[1] - box[3] / 2) * img_h)

    return x1, y1, x2, y2



def img_to_yolo_shape(img):

    # TODO: move
    yolo_image_height = 448
    yolo_image_width = 448

    img = image_utils.resize_image(img, yolo_image_height, yolo_image_width)

    return img

def yolo_cords(cords, image_shape):

    # TODO: move
    yolo_image_height = 448
    yolo_image_width = 448

    image_height, image_width, _ = image_shape

    ratio_height = yolo_image_height / image_height
    ratio_width = yolo_image_width / image_width

    new_cords = []
    for cord in cords:
        new_cord = (cord[0] * ratio_width, cord[1] * ratio_height,
                    cord[2] * ratio_width, cord[3] * ratio_height)
        new_cords.append(new_cord)

    return new_cords

def grid_index(x, y):

    # TODO: read from config
    yolo_image_width = 448
    yolo_image_height = 448

    grid_size = 7

    return int(x/(1/grid_size )), int(y/(1/grid_size))

