import tensorflow as tf
import numpy as np
from object_detection.model.base_model import ModelBase
from object_detection.model.cnn_model import CNNModel
from object_detection.config.config_reader import ConfigReader


class YOLO(ModelBase):

    def __init__(self, network: CNNModel, config: ConfigReader, anchors=None):
        super(YOLO, self).__init__(config)

        self.network = network

        self.boxes = None
        self.scores = None
        self.classes = None
        self.img = None

        # TODO: read from config
        #self.anchors = [(0.1, 0.15), (0.3, 0.25), (0.4, 0.5)]
        self.anchors = [[0.05524553571428571, 0.045619419642857144], [0.022042410714285716, 0.029296875], [0.13853236607142858, 0.10407366071428571]]

        self.init_placeholders()

    def init_placeholders(self):

        image_height = self.config.image_height()
        image_width = self.config.image_width()
        num_classes = self.config.num_classes()
        grid_size = self.config.grid_size()
        num_anchors = self.config.num_anchors()

        self.x = tf.placeholder(tf.float32, [None, image_height, image_width, 3])
        self.y = tf.placeholder(tf.float32, [None, grid_size, grid_size, num_anchors, 5 + num_classes])

        self.is_training = tf.placeholder(tf.bool, name='is_training')

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

        self.input = input

        x, y = input['x'], input['y']

        print('Building computation graph...')

        yolo_output = self.network.build_network(x, self.is_training)

        if len(yolo_output.get_shape()) > 5:
            raise Exception('Prediction Across Scale not supported.')

        yolo_output = self.transform_output_logits(yolo_output)

        self.loss = self.yolo_loss(yolo_output, y)

        self.opt = self.optimizer(self.loss, self.config.learning_rate())

        self.eval = self.yolo_eval(yolo_output)

    def transform_output_logits(self, logits):

        cords = tf.sigmoid(logits[..., 0:2])

        sizes = tf.exp(logits[..., 2:4])

        confidences = tf.sigmoid(logits[..., 4:5])

        classes = tf.nn.softmax(logits[..., 5:])

        return tf.concat([
            cords,
            sizes,
            confidences,
            classes
        ], axis=-1)

    def optimizer(self, loss, start_learning_rate=0.0001):

        # self.learning_rate = tf.cond(tf.math.greater(tf.constant(300), self.global_step_tensor),
        #             lambda: tf.constant(start_learning_rate),
        #             lambda: tf.train.polynomial_decay(0.0001, self.global_step_tensor - tf.constant(300), 50, 0.0000001))


        # learning_rate = tf.train.cosine_decay_restarts(start_learning_rate, self.global_step_tensor, first_decay_steps=100,
        #                              t_mul=2.0, m_mul=1.2, alpha=0.0, name=None)

        #self.learning_rate = tf.train.exponential_decay(self.config.learning_rate(), self.global_step_tensor, 50, 0.9, staircase=True)

        #tf.train.polynomial_decay(self.config.learning_rate(), self.global_step_tensor, 50, 0.0000001)

        self.learning_rate = tf.constant(start_learning_rate)

        print(start_learning_rate)

        tf.summary.scalar('learning_rate', self.learning_rate)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # TODO: control_dependencies added cause of batch_norm ?
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(loss, global_step=self.global_step_tensor)
        train_op = tf.group([train_op, update_ops])

        return train_op

    def yolo_loss(self, predict, label, lambda_cord=5, lambda_noobj=0.5):

        # INPUT: (?, grid_size, grid_size, num_anchors, 5 + num_classes)

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

    def yolo_eval(self, transformed_logits):

        with tf.variable_scope('eval'):
            boxes, scores, classes = self.eval_boxes(transformed_logits)

        return boxes, scores, classes

    def cord_loss(self, label, pred, lambda_cord=5):
        # INPUT: (?, grid_size, grid_size, num_anchors, 5 + num_classes)

        with tf.name_scope('cord'):
            # contains object(x>0) or does not contain object(x==0)
            cord_mask = tf.expand_dims(label[..., 4], axis=-1)

            self.mask = cord_mask

            pred_cord = pred[..., 0:2]
            label_cord = label[..., 0:2]

            pred_size = pred[..., 2:4]
            label_size = label[..., 2:4]

            # TODO: Add punishment?
            # hack from darknet: box punishment - give higher weights to small boxes
            # box_loss_scale = 2. - (label_size[..., 0] * label_size[..., 1])
            # box_loss_scale = tf.expand_dims(box_loss_scale, axis=-1)
            #
            # a = tf.constant(3.14, shape=box_loss_scale.get_shape(), dtype=tf.float32)
            # b = tf.constant(0, shape=box_loss_scale.get_shape(), dtype=tf.float32)
            # self.debug = tf.concat([label[..., 2:4], tf.where(tf.greater(0., box_loss_scale), a, b), box_loss_scale], axis=-1)

            # inverting ground truth to match network output for gradient update (pred_size is transformed using exp)
            pred_size = tf.math.log(pred_size)
            label_size = tf.math.log(label_size)
            # numerical stability cause of using log transform
            label_size = tf.where(tf.math.is_inf(label_size), tf.zeros_like(label_size), label_size)
            pred_size = tf.where(tf.math.is_inf(pred_size), tf.zeros_like(pred_size), pred_size)

            # calculate loss for x,y
            loss_cord = (cord_mask * tf.square(pred_cord - label_cord)) #* box_loss_scale
            loss_cord = tf.reduce_sum(loss_cord, axis=[1, 2, 3, 4]) * lambda_cord # TODO: test if should delete lambda?
            loss_cord = tf.reduce_mean(loss_cord)

            # calculate loss for w,h
            loss_size = (cord_mask * tf.square(pred_size - label_size)) #* box_loss_scale
            loss_size = tf.reduce_sum(loss_size, axis=[1, 2, 3, 4]) * lambda_cord # TODO: test if should delete lambda?
            loss_size = tf.reduce_mean(loss_size)

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

            # good detections
            object_detections = tf.cast(iou > 0.5, dtype=tf.float32)
            bad_object_detections = tf.cast(iou < 0.5, dtype=tf.float32)  # 1 - object_detections

            # NOT DETECTED OBJECT - punish bad detection
            #no_objects_loss = mask_noobj * tf.square(confidence_label - (confidence_pred * bad_object_detections)) * lambda_noobj
            no_objects_loss = mask_noobj * tf.square(confidence_label - confidence_pred) * bad_object_detections * lambda_noobj

            loss_noobj = tf.reduce_sum(no_objects_loss, axis=[1, 2, 3, 4]) * lambda_noobj
            loss_noobj = tf.reduce_mean(loss_noobj)

            # DETECTED - punish wrong detections
            #objects_loss = mask_obj * tf.square((confidence_label - (confidence_pred * object_detections))) * lambda_obj
            objects_loss = mask_obj * tf.square(confidence_label - confidence_pred) * lambda_obj

            loss_obj = tf.reduce_sum(objects_loss, axis=[1, 2, 3, 4])
            loss_obj = tf.reduce_mean(loss_obj)

        return loss_obj, loss_noobj

    def iou(self, label_box, pred_box):

        cord_pred, size_pred = tf.split(pred_box, 2, axis=-1)
        cord_true, size_true = tf.split(label_box, 2, axis=-1)

        image_size = self.config.image_width()
        cord_pred = cord_pred * image_size
        cord_true = cord_true * image_size

        size_pred = size_pred * self.anchors
        size_true = size_true * self.anchors

        pred_x, pred_y = tf.split(cord_pred, 2, axis=-1)
        pred_w, pred_h = tf.split(size_pred, 2, axis=-1)

        true_x, true_y = tf.split(cord_true, 2, axis=-1)
        true_w, true_h = tf.split(size_true, 2, axis=-1)

        # pred_x, pred_y, pred_w, pred_h = tf.split(pred_box, 4, axis=-1)
        # true_x, true_y, true_w, true_h = tf.split(label_box, 4, axis=-1)

        # pred_x, pred_y, pred_w, pred_h = tf.split(pred_box, 4, axis=1)
        # true_x, true_y, true_w, true_h = tf.split(label_box, 4, axis=1)

        # update boxes to absolute values
        # pred_w, pred_h = image_size*pred_w, image_size*pred_h
        # true_w, true_h = image_size*true_w, image_size*true_h
        #
        # pred_x, pred_y = 64*pred_x, 64*pred_y
        # true_x, true_y = 64*true_x, 64*true_y

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

        # adding 0.00001 for numerical stability
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
            loss_class = tf.reduce_sum(tf.square(loss_class), axis=[1, 2, 3, 4])
            #loss_class = mask_class * tf.nn.sigmoid_cross_entropy_with_logits(labels=class_label, logits=class_pred)

            loss_class = tf.reduce_mean(loss_class)

        return loss_class

    def eval_boxes(self, y_pred):

        boxes = y_pred[..., :4]
        confidence = tf.expand_dims(y_pred[..., 4], axis=-1)
        classes = y_pred[..., :5]

        batch_size = y_pred.get_shape()[0]

        # convert to from (x,y,w,h) to (y1, x1, y2, x2) cause of nms
        boxes = self.convert_to_min_max_cord(boxes, batch_size)

        #self.debug = [y_pred[..., :4], boxes]

        boxes, scores, classes = self.filter_boxes(boxes, confidence, classes)

        # TODO: revert box swapping after NMS
        return self.non_max_suppression(boxes, scores, classes)


    def filter_boxes(self, boxes, confidence, classes, threshold=0.5):
        """Filters YOLO boxes by thresholding on object and class confidence."""


        # Compute box scores
        box_scores = confidence * classes

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

        nms_index = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor,
                                                 iou_threshold=iou_threshold, score_threshold=score_threshold)

        # nms_index = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)

        boxes = tf.gather(boxes, nms_index)
        scores = tf.gather(scores, nms_index)
        classes = tf.gather(classes, nms_index)

        return boxes, scores, classes

    def convert_to_min_max_cord(self, boxes, batch_size):

        box_xy = boxes[..., 0:2]
        box_wh = boxes[..., 2:4]

        # number of grids
        num_grids = tf.constant(self.config.grid_size())

        # create grid (B, 7, 7, A, 2) where we have the min absolute coordinates of each grid
        grid = tf.meshgrid(tf.range(num_grids), tf.range(num_grids), indexing='ij')
        grid = tf.stack(grid, axis=-1)
        grid = tf.expand_dims(grid, axis=2)
        grid = tf.tile(grid, [1, 1, len(self.anchors), 1])
        grid = tf.cast(grid, dtype=tf.float32)

        image_size = tf.constant(self.config.image_height(), dtype=tf.float32)
        cell_size = tf.constant(self.config.image_height()/self.config.grid_size(), dtype=tf.float32)
        grid = tf.stack([grid] * batch_size)

        # calculate relative coordinates, where xy is relative to cell_sizes and wh to appropriate anchor
        #box_xy = (box_xy * cell_size) + grid
        box_xy = box_xy + grid
        box_wh = (box_wh * np.array(self.anchors))

        # convert to absolute cordinates in px
        grid_length = cell_size * tf.cast(num_grids, dtype=tf.float32)
        box_xy = box_xy * cell_size
        box_wh = box_wh * grid_length

        # convert to min max coordinates
        box_mins = box_xy - (box_wh / 2.)
        box_maxes = box_xy + (box_wh / 2.)

        # to relative sizes
        # grid_length = cell_size * tf.cast(num_grids, dtype=tf.float32)
        # box_mins = box_mins/grid_length
        # box_maxes = box_maxes/grid_length

        # TODO: check if swapping needed for NMS
        # boxes = tf.concat([box_mins[..., 1:2],  # y_min
        #                    box_mins[..., 0:1],  # x_min
        #                    box_maxes[..., 1:2],  # y_max
        #                    box_maxes[..., 0:1]  # x_max
        # ], axis=-1)

        boxes = tf.concat([box_mins[..., 0:1],   # x_min
                           box_mins[..., 1:2],   # y_min
                           box_maxes[..., 0:1],  # x_max
                           box_maxes[..., 1:2],  # y_max
        ], axis=-1)

        self.debug = boxes

        # return flatten out boxes (B*7*7, 4)
        #return tf.reshape(boxes, (-1, 4))

        return boxes



if __name__ == '__main__':

    #sess = tf.InteractiveSession()

    # grid = tf.meshgrid(tf.range(7), tf.range(7))
    # grid = tf.stack(grid, axis=-1)
    #
    # cell_size = tf.constant(int(448 / 7), dtype=tf.int32)
    #
    # grid = grid * cell_size
    #
    # print(grid.eval())
    # print(grid.get_shape())
    # num_classes = 5
    #
    # b = 1
    #
    # y_pred = tf.convert_to_tensor(np.random.rand(8, 7, 7, (b*5) + num_classes), np.float32)
    # y_true = tf.convert_to_tensor(np.random.rand(8, 7, 7, (b*5) + num_classes), np.float32)

    # test = tf.expand_dims(y_true[..., 4], axis=-1)
    #
    # print(test.eval())
    #
    # test2 = y_true[..., 4]
    #
    # print(test2.eval())

    # config_path = '/Users/adam.zvada/Documents/Dev/object-detection/config/test.yml'
    #
    # yolo = YOLO(ConfigReader(config_path))
    #
    # # a = [[0.5, 0.5, 64/448, 64/448], [0.0, 0.0, 64/448, 64/448]]
    # # b = [[1.0, 1.0, 64/448, 64/448], [1.0, 1.0, 64/448, 64/448]]
    #
    # a = [[0.5, 0.5, 1., 1.], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.25, 0.25]]
    # b = [[0.5, 0.5, 1., 1.], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.25, 0.25]]
    #
    # label_box = tf.convert_to_tensor(a, np.float32)
    # pred_box = tf.convert_to_tensor(b, np.float32)

    # test IOU
    #print(yolo.iou(label_box, pred_box).eval())

    #print(yolo.convert_to_min_max_cord(label_box, pred_box).eval())

    # print(yolo.yolo_loss(predict=y_pred, label=y_true).eval())

    # ----------------CAN BE USED FOR TESTS----------------------------
    # ---------------TEST---coordinates conversion---------------------
    from object_detection.dataset.udacity_object_dataset import UdacityObjectDataset
    from object_detection.dataset.image_iterator import ImageIterator
    import sys
    import numpy
    numpy.set_printoptions(threshold=sys.maxsize)

    sess = tf.InteractiveSession()

    config_path = '/Users/adam.zvada/Documents/Dev/object-detection/config/test.yml'

    yolo = YOLO(ConfigReader(config_path))

    dataset = UdacityObjectDataset(ConfigReader(config_path))
    iterator = ImageIterator(sess, yolo, dataset, ConfigReader(config_path))

    input, test_handle = iterator.create_iterator(mode='train')

    label = sess.run(input['y'], feed_dict={iterator.handle_placeholder: test_handle})

    tf_label = tf.convert_to_tensor(label, np.float32)

    for g_x in range(0,7):
        for g_y in range(0, 7):
            for i, (rel_anchor_width, rel_anchor_height) in enumerate(yolo.anchors):
                cell_x = label[0][g_x][g_y][i][0]
                cell_y = label[0][g_x][g_y][i][1]
                a_w = label[0][g_x][g_y][i][2]
                a_h = label[0][g_x][g_y][i][3]

                label[0][g_x][g_y][i][0] = int((((cell_x * 64) + g_x * 64) - (a_w * (rel_anchor_width * 448) / 2)))
                label[0][g_x][g_y][i][1] = int((((cell_y * 64) + g_y * 64) - (a_h * (rel_anchor_height * 448) / 2)))

                label[0][g_x][g_y][i][2] = int((((cell_x * 64) + g_x * 64) + (a_w * (rel_anchor_width * 448) / 2)))
                label[0][g_x][g_y][i][3] = int((((cell_y * 64) + g_y * 64) + (a_h * (rel_anchor_height * 448) / 2)))

    print(label[0])

    dataset.df_true['x_min'] = dataset.df_true['x_min'] * 448/1920
    dataset.df_true['y_min'] = dataset.df_true['y_min'] * 448/1200
    dataset.df_true['x_max'] = dataset.df_true['x_max'] * 448/1920
    dataset.df_true['y_max'] = dataset.df_true['y_max'] * 448/1200

    print(dataset.df_true[['x_min', 'y_min', 'x_max', 'y_max']])

    print(yolo.convert_to_min_max_cord(tf_label, 8).eval()[0])

    # ---------------TEST---coordinates conversion---------------------

