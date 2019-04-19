import tensorflow as tf
import numpy as np
from object_detection.model.cnn_model import CNNModel
from object_detection.config.config_reader import ConfigReader


class YOLO(CNNModel):

    def __init__(self, config: ConfigReader):
        super(YOLO, self).__init__(config)

        self.boxes = None
        self.scores = None
        self.classes = None
        self.img = None

        self.init_placeholders()

    def init_placeholders(self):

        image_height = self.config.image_height()
        image_width = self.config.image_width()
        num_classes = self.config.num_classes()
        grid_size = self.config.grid_size()
        boxes_per_cell = self.config.boxes_per_cell()

        self.x = tf.placeholder(tf.float32, [None, image_height, image_width, 3])
        self.y = tf.placeholder(tf.float32, [None, grid_size, grid_size, (boxes_per_cell*5) + num_classes])

    def get_tensor_boxes(self):
        return self.boxes

    def get_tensor_scores(self):
        return self.scores

    def get_tensor_classes(self):
        return self.classes

    def get_optimizer(self):
        return self.opt

    def get_loss(self):
        return self.loss

    def get_image(self):
        return self.img

    def get_labels(self):
        return self.labels

    def build_model(self, input):

        x, y = input['x'], input['y']

        print('Building computation graph...')

        yolo_output = self.yolo(x)

        self.loss = self.yolo_loss(yolo_output, y)

        self.opt = self.optimizer(self.loss, self.config.learning_rate())

        with tf.variable_scope('eval'):
            boxes, scores, classes = self.eval_boxes(yolo_output)

            self.boxes = boxes
            self.scores = scores
            self.classes = classes

            self.img = x

            self.labels = y

    def yolo(self, input):
        print('Input: {}'.format(input.get_shape()))

        # LAYER 1
        conv = self.conv(input, filter_height=7, filter_width=7, num_filters=64,
                          stride_x=2, stride_y=2, padding='SAME', scope_name='conv1')
        print('Conv1: {}'.format(conv.get_shape()))

        pool = self.max_pool(conv, filter_height=2, filter_width=2, stride_x=2, stride_y=2,
                              padding='VALID', scope_name='pool1')
        print('Pool1: {}'.format(pool.get_shape()))

        # LAYER 2
        conv = self.conv(pool, filter_height=3, filter_width=3, num_filters=192,
                          stride_x=1, stride_y=1, padding='SAME', scope_name='conv2')
        print('Conv2: {}'.format(conv.get_shape()))

        pool = self.max_pool(conv, filter_height=2, filter_width=2, stride_x=2, stride_y=2,
                              padding='VALID', scope_name='pool1')
        print('Pool1: {}'.format(pool.get_shape()))

        # LAYER 3
        conv = self.conv(pool, filter_height=1, filter_width=1, num_filters=128,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv3')
        print('Conv3: {}'.format(conv.get_shape()))

        # LAYER 4
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=256,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv4')
        print('Conv4: {}'.format(conv.get_shape()))

        # LAYER 5
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=256,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv5')
        print('Conv5: {}'.format(conv.get_shape()))

        # LAYER 6
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv6')
        print('Conv6: {}'.format(conv.get_shape()))

        pool = self.max_pool(conv, filter_height=2, filter_width=2, stride_x=2, stride_y=2,
                             padding='VALID', scope_name='pool3')
        print('Pool3: {}'.format(pool.get_shape()))

        # LAYER 7
        conv = self.conv(pool, filter_height=1, filter_width=1, num_filters=256,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv7')
        print('Conv7: {}'.format(conv.get_shape()))

        # LAYER 8
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv8')
        print('Conv8: {}'.format(conv.get_shape()))

        # LAYER 9
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=256,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv9')
        print('Conv9: {}'.format(conv.get_shape()))

        # LAYER 10
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv10')
        print('Conv10: {}'.format(conv.get_shape()))

        # LAYER 11
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=256,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv11')
        print('Conv11: {}'.format(conv.get_shape()))

        # LAYER 12
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv12')
        print('Conv12: {}'.format(conv.get_shape()))

        # LAYER 13
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv13')
        print('Conv13: {}'.format(conv.get_shape()))

        # LAYER 14
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=512,
                          stride_x=1, stride_y=1, padding='SAME', scope_name='conv14')
        print('Conv14: {}'.format(conv.get_shape()))

        # LAYER 15
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv15')
        print('Conv15: {}'.format(conv.get_shape()))

        # LAYER 16
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                          stride_x=1, stride_y=1, padding='SAME', scope_name='conv16')
        print('Conv16: {}'.format(conv.get_shape()))

        pool = self.max_pool(conv, filter_height=2, filter_width=2, stride_x=2, stride_y=2,
                             padding='VALID', scope_name='pool4')
        print('Pool4: {}'.format(pool.get_shape()))

        # LAYER 17
        conv = self.conv(pool, filter_height=1, filter_width=1, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv17')
        print('Conv17: {}'.format(conv.get_shape()))

        # LAYER 18
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv18')
        print('Conv18: {}'.format(conv.get_shape()))

        # LAYER 19
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv19')
        print('Conv19: {}'.format(conv.get_shape()))

        # LAYER 20
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv20')
        print('Conv20: {}'.format(conv.get_shape()))

        # LAYER 21
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv21')
        print('Conv21: {}'.format(conv.get_shape()))

        # LAYER 22
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=2, stride_y=2, padding='SAME', scope_name='conv22')
        print('Conv22: {}'.format(conv.get_shape()))

        # LAYER 23
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv23')
        print('Conv23: {}'.format(conv.get_shape()))

        # LAYER 24
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv24')
        print('Conv24: {}'.format(conv.get_shape()))

        # flatten cnn output for fully connected layer
        feature_dim = conv.get_shape()[1:4].num_elements()
        cnn_output = tf.reshape(conv, [self.config.batch_size(), feature_dim])
        print('Flatten: {}'.format(cnn_output.get_shape()))

        # LAYER 6
        fc1 = self.fully_connected(cnn_output, 4096, scope_name='fc6')
        print('FC1: {}'.format(fc1.get_shape()))

        # LAYER 7 - OUTPUT
        grid_size = self.config.grid_size()
        num_classes = self.config.num_classes()
        boxes_per_cell = self.config.boxes_per_cell()

        output_size = (grid_size*grid_size) * (boxes_per_cell*5 + num_classes)

        fc2 = self.fully_connected(fc1, output_size , scope_name='fc7')
        print('FC2: {}'.format(fc2.get_shape()))

        logits = tf.reshape(fc2, shape=[self.config.batch_size(), grid_size, grid_size, boxes_per_cell*5 + num_classes])
        print('Logits: {}'.format(logits.get_shape()))

        logits = self.transform_output_logits(logits)

        return logits

    def transform_output_logits(self, logits):

        #cords = tf.sigmoid(logits[..., 0:2])
        cords = logits[..., 0:2]

        sizes = logits[..., 2:4]
        #sizes = logits[..., 2:4]

        #self.debug = logits[..., 2:4]

        #confidences = tf.sigmoid(logits[..., 4:5])
        confidences = logits[..., 4:5]

        #classes = tf.nn.softmax(logits[..., 5:])
        classes = logits[..., 5:]

        return tf.concat([
            cords,
            sizes,
            confidences,
            classes
        ], axis=-1)

    def optimizer(self, loss, start_learning_rate=0.0001):


        learning_rate = tf.train.cosine_decay_restarts(start_learning_rate, self.global_step_tensor, first_decay_steps=100,
                                       t_mul=2.0, m_mul=1.2, alpha=0.0, name=None)

        #learning_rate = tf.train.exponential_decay(self.config.learning_rate(), self.global_step_tensor, 50, 0.5, staircase=True)

        self.learning_rate = learning_rate

        tf.summary.scalar('learning_rate', learning_rate)

        opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        return opt

    def yolo_loss(self, predict, label, lambda_cord=5, lambda_noobj=0.5):

        # INPUT: (?, grid_size, grid_size, bB*5 + num_classes)

        # TODO: we ASSUME B = 1

        with tf.name_scope('loss'):
            loss_size, loss_cord = self.cord_loss(label, predict, lambda_cord)
            self.loss_cord = loss_cord
            self.loss_size = loss_size

            tf.summary.scalar('loss_cord', loss_cord)
            tf.summary.scalar('loss_size', loss_size)

            loss_obj, loss_noobj = self.confidence_loss(label, predict, lambda_noobj)
            self.loss_obj = loss_obj
            self.loss_noobj = loss_noobj

            tf.summary.scalar('loss_obj', loss_obj)
            tf.summary.scalar('loss_noobj', loss_noobj)

            loss_confidence = loss_obj + loss_noobj

            loss_class = self.classes_loss(label, predict)
            self.loss_class = loss_class

            tf.summary.scalar('loss_class', loss_class)

            loss = tf.add(loss_cord, loss_size)
            loss = tf.add(loss, loss_confidence)
            loss = tf.add(loss, loss_class)

            tf.summary.scalar('loss', loss)

        return loss

    def cord_loss(self, label, pred, lambda_cord=5):
        # INPUT: (?, grid_size, grid_size, B*5 + num_classes)

        # TODO: we ASSUME B = 1

        # always is 0 or 1

        #indicator_coord = tf.math.ceil(indicator_coord)

        # ---TESTING PURPOSE----
        # indicator_coord = tf.dtypes.cast(indicator_coord > 0.4, tf.float32)
        # print(indicator_coord.eval())

        with tf.name_scope('cord'):
            # contains object(x>0) or does not contain object(x==0)
            cord_mask = tf.expand_dims(label[..., 4], axis=-1)

            self.mask = cord_mask

            pred_cord = pred[..., 0:2]
            label_cord = label[..., 0:2]

            pred_size = pred[..., 2:4]
            label_size = label[..., 2:4]

            loss_cord = cord_mask * tf.square(pred_cord - label_cord)

            loss_cord = tf.reduce_sum(loss_cord, axis=[1, 2, 3]) * lambda_cord
            loss_cord = tf.reduce_mean(loss_cord)

            loss_size = cord_mask * (tf.sqrt(tf.abs(pred_size)) - tf.sqrt(label_size))
            #loss_size = cord_mask * (pred_size - label_size)
            loss_size = tf.reduce_sum(tf.square(loss_size), axis=[1, 2, 3]) * lambda_cord
            loss_size = tf.reduce_mean(loss_size)

            # loss_cord = tf.multiply(indicator_coord, tf.squared_difference(label_cord, pred_cord))
            # loss_cord = tf.reduce_sum(loss_cord)
            #
            # loss_size = tf.multiply(indicator_coord, tf.squared_difference(label_size, pred_size))
            # loss_size = tf.reduce_sum(loss_size)

        return loss_size, loss_cord

    def confidence_loss(self, label, pred, lambda_obj=1, lambda_noobj=0.5):

        # ---TESTING PURPOSE----
        # indicator_noobj = tf.dtypes.cast(indicator_noobj > 0.6, tf.float32)
        # print(indicator_noobj.eval())
        # ---TESTING PURPOSE----


        # ---TESTING PURPOSE----
        # indicator_obj = tf.dtypes.cast(indicator_obj > 0.4, tf.float32)
        # print(indicator_obj.eval())
        # ---TESTING PURPOSE----

        with tf.name_scope('confidence'):
            mask_obj = tf.expand_dims(label[..., 4], axis=-1)
            mask_noobj = (1 - mask_obj)

            confidence_label = tf.expand_dims(label[..., 4], axis=-1)
            confidence_pred = tf.expand_dims(pred[..., 4], axis=-1)

            iou = self.iou(label[..., 0:4], pred[..., 0:4])
            self.debug = iou

            # good detections
            object_detections = tf.cast(iou > 0.5, dtype=tf.float32)
            bad_object_detections = tf.cast(iou < 0.5, dtype=tf.float32)  # 1 - object_detections

            # NOT DETECTED OBJECT - punish bad detection
            #no_objects_loss = mask_noobj * tf.square(confidence_label - (confidence_pred * bad_object_detections)) * lambda_noobj
            no_objects_loss = mask_noobj * tf.square(confidence_label - confidence_pred) * bad_object_detections * lambda_noobj

            loss_noobj = tf.reduce_sum(no_objects_loss, axis=[1, 2, 3]) * lambda_noobj
            loss_noobj = tf.reduce_mean(loss_noobj)

            # DETECTED - punish wrong detections
            #objects_loss = mask_obj * tf.square((confidence_label - (confidence_pred * object_detections))) * lambda_obj
            objects_loss = mask_obj * tf.square(confidence_label - confidence_pred) * lambda_obj

            loss_obj = tf.reduce_sum(objects_loss, axis=[1, 2, 3])
            loss_obj = tf.reduce_mean(loss_obj)

        return loss_obj, loss_noobj

    def iou(self, label_box, pred_box):

        pred_x, pred_y, pred_w, pred_h = tf.split(pred_box, 4, axis=3)
        true_x, true_y, true_w, true_h = tf.split(label_box, 4, axis=3)

        # TODO: have width and height relative to the grid cell from dataset
        pred_w, pred_h = (448*pred_w)*(1/64), (448*pred_h)*(1/64)
        true_w, true_h = (448*true_w)*(1/64), (448*true_h)*(1/64)

        # from to (x, y, w, h) to (x_min, y_min, x_max, y_max)
        pred_x_min, pred_y_min = pred_x - pred_w / 2, pred_y - pred_h / 2
        pred_x_max, pred_y_max = pred_x + pred_h / 2, pred_y + pred_h / 2

        true_x_min, true_y_min = true_x - true_w / 2, true_y - true_h / 2
        true_x_max, true_y_max = true_x + true_w / 2, true_y + true_h / 2

        # get cords of intersect box
        inter_x_min = tf.maximum(pred_x_min, true_x_min)
        inter_y_min = tf.maximum(pred_y_min, true_y_min)

        inter_x_max = tf.minimum(pred_x_max, true_x_max)
        inter_y_max = tf.minimum(pred_y_max, true_y_max)

        # get area of intersect box, true box and pred box
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        true_area = (true_x_max - true_x_min) * (true_y_max - true_y_min)
        pred_area = (pred_x_max - pred_x_min) * (pred_y_max - pred_y_min)

        # print(inter_area.eval())
        # print(true_area.eval())
        # print(pred_area.eval())

        # union of area coverd by true and pred boxes
        union = true_area + pred_area - inter_area

        # add 0.00001 for numerical stability
        return inter_area / (union + 0.0001)

    def classes_loss(self, label, pred):

        # ---TESTING PURPOSE----
        # indicator_class = tf.dtypes.cast(indicator_class > 0.6, tf.float32)
        # print(indicator_class.eval())
        # ---TESTING PURPOSE----

        with tf.name_scope('class'):
            mask_class = tf.expand_dims(label[..., 4], axis=-1)

            class_label = label[..., 5:]
            class_pred = pred[..., 5:]

            loss_class = mask_class * (class_pred - class_label)
            loss_class = tf.reduce_sum(tf.square(loss_class), axis=[1, 2, 3])
            loss_class = tf.reduce_mean(loss_class)

        return loss_class

    def eval_boxes(self, y_pred):

        boxes, scores, classes = self.filter_boxes(y_pred)

        return self.non_max_suppression(boxes, scores, classes)


    def filter_boxes(self, y_pred, threshold=0.5):
        """Filters YOLO boxes by thresholding on object and class confidence."""

        box_confidence = tf.expand_dims(y_pred[..., 4], axis=-1)
        boxes = y_pred[..., :4]
        box_class_probs = y_pred[..., :5]

        # Compute box scores
        box_scores = box_confidence * box_class_probs

        # index of highest box score (return vector?)
        box_classes = tf.argmax(box_scores, axis=-1)

        # value of the highest box score (return vector?)
        box_class_scores = tf.reduce_max(box_scores, axis=-1)

        prediction_mask = (box_class_scores >= threshold)

        boxes = tf.boolean_mask(boxes, prediction_mask)
        scores = tf.boolean_mask(box_class_scores, prediction_mask)
        classes = tf.boolean_mask(box_classes, prediction_mask)

        return boxes, scores, classes

    def non_max_suppression(self, boxes, scores, classes, max_boxes=30, score_threshold=0.3, iou_threshold=0.5):
        """ Applying NMS, optimized box location for same classes"""

        max_boxes_tensor = tf.constant(max_boxes, dtype='int32')

        # convert to from (x,y,w,h) to (y1, x1, y2, x2) cause of nms
        boxes = self.convert_to_min_max_cord(boxes)

        nms_index = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor,
                                                 iou_threshold=iou_threshold, score_threshold=score_threshold)

        boxes = tf.gather(boxes, nms_index)
        scores = tf.gather(scores, nms_index)
        classes = tf.gather(classes, nms_index)

        return boxes, scores, classes

    def convert_to_min_max_cord(self, boxes):
        box_xy = boxes[..., 0:2]
        box_wh = boxes[..., 2:4]

        box_mins = box_xy - (box_wh / 2.)
        box_maxes = box_xy + (box_wh / 2.)

        return tf.concat([
            box_mins[..., 1:2],  # y_min
            box_mins[..., 0:1],  # x_min
            box_maxes[..., 1:2],  # y_max
            box_maxes[..., 0:1]  # x_max
        ], axis=-1)



if __name__ == '__main__':

    sess = tf.InteractiveSession()

    num_classes = 5
    b = 1

    y_pred = tf.convert_to_tensor(np.random.rand(8, 7, 7, (b*5) + num_classes), np.float32)
    y_true = tf.convert_to_tensor(np.random.rand(8, 7, 7, (b*5) + num_classes), np.float32)

    # test = tf.expand_dims(y_true[..., 4], axis=-1)
    #
    # print(test.eval())
    #
    # test2 = y_true[..., 4]
    #
    # print(test2.eval())

    config_path = '/Users/adam.zvada/Documents/Dev/object-detection/config/test.yml'

    yolo = YOLO(ConfigReader(config_path))

    # a = [[0.5, 0.5, 64/448, 64/448], [0.0, 0.0, 64/448, 64/448]]
    # b = [[1.0, 1.0, 64/448, 64/448], [1.0, 1.0, 64/448, 64/448]]

    a = [[0.0, 0.0, 64/448, 64/448]]
    b = [[1.0, 1.0, 64/448, 64/448]]


    label_box = tf.convert_to_tensor(a, np.float32)
    pred_box = tf.convert_to_tensor(b, np.float32)

    # test IOU
    print(yolo.iou(label_box, pred_box).eval())

    # print(yolo.yolo_loss(predict=y_pred, label=y_true).eval())
