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
        self.detections = None

        # TODO: read from config
        #self.anchors = [(0.1, 0.15), (0.3, 0.25), (0.4, 0.5)]
        anchors_large = [[161.19999695, 226.4888916], [57.19999695, 97.06668091], [19.82500076, 63.41111755]]
        anchors_large = np.array(anchors_large)
        anchors_large /= self.config.image_width()

        anchors_medium = [[38.56666565, 32.58666992], [17.0625, 19.5], [8.93748474, 33.27999878]]
        anchors_medium = np.array(anchors_medium)
        anchors_medium /= self.config.image_width()

        anchors_small = [[5.6333313, 17.33332825], [9.5874939, 10.1111145], [4.0625, 7.80000305]]
        anchors_small = np.array(anchors_small)
        anchors_small /= self.config.image_width()

        self.anchors = [anchors_small, anchors_medium, anchors_large]

        self.init_placeholders()

    def init_placeholders(self):

        image_height = self.config.image_height()
        image_width = self.config.image_width()
        num_classes = self.config.num_classes()
        num_anchors = self.config.num_anchors()

        num_cells_small = self.config.num_cells_small()  # 52x52
        num_cells_medium = self.config.num_cells_medium()  # 26x26
        num_cells_large = self.config.num_cells_large()  # 13x13

        self.images = tf.placeholder(tf.float32, [None, image_height, image_width, 3])
        self.image_paths = tf.placeholder(tf.string, [None])

        self.y_small = tf.placeholder(tf.float32, [None, num_cells_small, num_cells_small, num_anchors, 5 + num_classes])
        self.y_medium = tf.placeholder(tf.float32, [None, num_cells_medium, num_cells_medium, num_anchors, 5 + num_classes])
        self.y_large = tf.placeholder(tf.float32, [None, num_cells_large, num_cells_large, num_anchors, 5 + num_classes])
        self.test_large = tf.placeholder(tf.float32, [None, num_cells_large, num_cells_large, num_anchors, 5 + num_classes])
        self.y = [self.y_small, self.y_medium, self.y_large]

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

    def label_to_boxes(self):
        """
        Test function identity transformation
        :return:
        """
        return self.eval_prediction(self.y_medium)

    def test_iou(self):
        """
        Test function identity transformation
        :return:
        """
        return self.iou(self.y_large[..., 0:4], self.y_large[..., 0:4], self.anchors[2])

    def build_model(self, mode='train', x=None, y=None):
        # TODO: add eval option
        if mode == 'train':
            if x is None or y is None:
                raise Exception('Missing input for building graph for training the model.')
            self.build_training_model(x, y)
        elif mode == 'predict':
            self.build_predict_model(self.images)
        else:
            raise Exception('Unsupported mode...')

    def build_training_model(self, x, y):

        logits = self.network.build_network(x, self.is_training)

        self.loss = 0
        # number of anchors defined number of scales
        for i, y_scale in enumerate(y):
            tf.is_nan(y_scale)
            transformed_logits = self.transform_output_logits(logits[i])

            self.loss += self.yolo_loss(predict=transformed_logits, label=y_scale, anchors=self.anchors[i])

        self.opt = self.optimizer(self.loss, self.config.learning_rate())

    def build_predict_model(self, images):

        logits = self.network.build_network(images, self.is_training)

        # number of anchors defined number of scales
        transformed_logits = [self.transform_output_logits(logits[i]) for i in range(self.config.num_scales())]

        self.detections = self.eval_prediction(transformed_logits)

    def transform_output_logits(self, logits):

        cords = tf.sigmoid(logits[..., 0:2])

        sizes = tf.exp(logits[..., 2:4])

        confidences = tf.sigmoid(logits[..., 4:5])

        classes = tf.sigmoid(logits[..., 5:])

        return tf.concat([
            cords,
            sizes,
            confidences,
            classes
        ], axis=-1)

    def optimizer(self, loss, start_learning_rate=0.01):

        stabel_lr_epoches = 80

        # self.learning_rate = tf.cond(
        #     tf.math.greater(tf.constant(stabel_lr_epoches * self.config.batch_size() * self.config.num_iterations()), self.global_step_tensor),
        #             lambda: tf.constant(start_learning_rate),
        #             lambda: tf.train.polynomial_decay(start_learning_rate, self.global_step_tensor - tf.constant(300), 50, 0.0000001))

        self.learning_rate = tf.constant(start_learning_rate)

        tf.summary.scalar('learning_rate', self.learning_rate)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # control_dependencies added cause of batch_norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(loss, global_step=self.global_step_tensor)
        train_op = tf.group([train_op, update_ops])

        return train_op

    def yolo_loss(self, predict, label, anchors, lambda_cord=5, lambda_noobj=0.5):

        # INPUT: (?, grid_size, grid_size, num_anchors, 5 + num_classes)

        with tf.name_scope('loss'):
            loss_size, loss_cord = self.cord_loss(label, predict, lambda_cord)
            self.loss_cord = loss_cord
            self.loss_size = loss_size

            tf.summary.scalar('loss_cord', loss_cord)
            tf.summary.scalar('loss_size', loss_size)

            loss_obj, loss_noobj = self.confidence_loss(label, predict, anchors, lambda_noobj)
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

    def cord_loss(self, label, pred, anchors, lambda_cord=5):
        # INPUT: (?, grid_size, grid_size, num_anchors, 5 + num_classes)

        with tf.name_scope('cord'):
            # contains object(x>0) or does not contain object(x==0)
            cord_mask = tf.expand_dims(label[..., 4], axis=-1)

            self.mask = cord_mask

            pred_cord = pred[..., 0:2]
            label_cord = label[..., 0:2]

            pred_size = pred[..., 2:4]
            label_size = label[..., 2:4]

            # hack from darknet: box punishment - give higher weights to small boxes
            label_size_absolute = label_size / anchors
            box_loss_scale = 2. - (label_size_absolute[..., 0] * label_size_absolute[..., 1])
            #box_loss_scale = tf.Print(box_loss_scale, [box_loss_scale], summarize=-1)
            box_loss_scale = tf.expand_dims(box_loss_scale, axis=-1)

            # inverting ground truth to match network output for gradient update (pred_size is transformed using exp)
            pred_size = tf.math.log(tf.clip_by_value(pred_size, 1e-4, 1e4))
            label_size = tf.math.log(tf.clip_by_value(label_size, 1e-4, 1e4))
            # numerical stability cause of using log transform
            label_size = tf.where(tf.math.is_inf(label_size), tf.zeros_like(label_size), label_size)
            pred_size = tf.where(tf.math.is_inf(pred_size), tf.zeros_like(pred_size), pred_size)

            # calculate loss for x,y
            loss_cord = (cord_mask * tf.square(pred_cord - label_cord)) #* box_loss_scale
            loss_cord = tf.reduce_sum(loss_cord, axis=[1, 2, 3, 4]) #* lambda_cord #
            loss_cord = tf.reduce_mean(loss_cord)

            # calculate loss for w,h
            loss_size = (cord_mask * tf.square(pred_size - label_size)) #* box_loss_scale
            loss_size = tf.reduce_sum(loss_size, axis=[1, 2, 3, 4]) #* lambda_cord #
            loss_size = tf.reduce_mean(loss_size)

        return loss_size, loss_cord

    def confidence_loss(self, label, pred, anchors, lambda_obj=1, lambda_noobj=0.5):

        with tf.name_scope('confidence'):
            mask_obj = tf.expand_dims(label[..., 4], axis=-1)
            mask_noobj = (1 - mask_obj)

            confidence_label = tf.expand_dims(label[..., 4], axis=-1)
            confidence_pred = tf.expand_dims(pred[..., 4], axis=-1)

            iou = self.iou(label[..., 0:4], pred[..., 0:4], anchors)
            iou = tf.reduce_max(iou, axis=-1)
            bad_object_detections = tf.cast(iou < 0.4, dtype=tf.float32)  # 1 - object_detections
            bad_object_detections = tf.expand_dims(bad_object_detections, axis=-1)

            # NOT DETECTED OBJECT - punish bad detection
            #no_objects_loss = mask_noobj * tf.square(confidence_label - (confidence_pred * bad_object_detections)) * lambda_noobj
            #no_objects_loss = mask_noobj * tf.square(confidence_label - confidence_pred) * bad_object_detections * lambda_noobj
            no_objects_loss = tf.keras.backend.binary_crossentropy(target=confidence_label, output=confidence_pred)
            no_objects_loss = mask_noobj * bad_object_detections * no_objects_loss

            # DETECTED - punish wrong detections
            #objects_loss = mask_obj * tf.square((confidence_label - (confidence_pred * object_detections))) * lambda_obj
            #objects_loss = mask_obj * tf.square(confidence_label - confidence_pred) #* lambda_obj
            objects_loss = tf.keras.backend.binary_crossentropy(target=confidence_label, output=confidence_pred)
            objects_loss = mask_obj * no_objects_loss

            loss_noobj = tf.reduce_sum(no_objects_loss, axis=[1, 2, 3, 4])
            loss_noobj = tf.reduce_mean(loss_noobj)

            loss_obj = tf.reduce_sum(objects_loss, axis=[1, 2, 3, 4])
            loss_obj = tf.reduce_mean(loss_obj)

        return loss_obj, loss_noobj

    def iou(self, label_box, pred_box, anchors):

        # convert from (x, y, w, h ) to (x_min, y_min, x_max, y_max) with respect to anchors
        label_box_minmax = self.convert_to_min_max_cord(label_box, anchors)
        pred_box_minmax = self.convert_to_min_max_cord(pred_box, anchors)

        pred_x_min, pred_y_min = pred_box_minmax[..., 0:1], pred_box_minmax[..., 1:2]
        pred_x_max, pred_y_max = pred_box_minmax[..., 2:3], pred_box_minmax[..., 3:4]

        true_x_min, true_y_min = label_box_minmax[..., 0:1], label_box_minmax[..., 1:2]
        true_x_max, true_y_max = label_box_minmax[..., 2:3], label_box_minmax[..., 3:4]

        # get cords of intersect box
        inter_x_min = tf.maximum(pred_x_min, true_x_min)
        inter_y_min = tf.maximum(pred_y_min, true_y_min)

        inter_x_max = tf.minimum(pred_x_max, true_x_max)
        inter_y_max = tf.minimum(pred_y_max, true_y_max)

        inter_w = tf.maximum(inter_x_max - inter_x_min, 0.0)
        inter_h = tf.maximum(inter_y_max - inter_y_min, 0.0)

        # get area of intersect box, true box and pred box
        inter_area = inter_w * inter_h
        true_area = (true_x_max - true_x_min) * (true_y_max - true_y_min)
        pred_area = (pred_x_max - pred_x_min) * (pred_y_max - pred_y_min)

        # union of area covered by true and pred boxes
        union = true_area + pred_area - inter_area

        # adding 0.00001 for numerical stability
        return inter_area / (union + 0.0001)

    def classes_loss(self, label, pred):

        with tf.name_scope('class'):
            mask_class = tf.expand_dims(label[..., 4], axis=-1)

            class_label = label[..., 5:]
            class_pred = pred[..., 5:]

            # loss_class = mask_class * (class_pred - class_label)
            # loss_class = tf.reduce_sum(tf.square(loss_class), axis=[1, 2, 3, 4])
            loss_class = mask_class * tf.keras.backend.binary_crossentropy(target=class_label, output=class_pred)

            loss_class = tf.reduce_sum(loss_class, axis=[1, 2, 3, 4])
            loss_class = tf.reduce_mean(loss_class)

        return loss_class

    def eval_prediction(self, y_pred):

        boxes = y_pred[..., :4]
        confidence = y_pred[..., 4:5]
        classes = y_pred[..., 5:]

        # convert to from (x,y,w,h) to (y1, x1, y2, x2) cause of nms
        boxes = self.convert_to_min_max_cord(boxes, anchors=self.anchors[1])  # TODO: support multiscale anchors!

        boxes, scores, classes = self.filter_boxes(boxes, confidence, classes)

        return boxes, scores, classes


    def filter_boxes(self, boxes, confidence, classes, threshold=0.5):
        """Filters YOLO boxes by thresholding on object and class confidence."""

        # Compute box scores
        box_scores = confidence * classes

        # index of highest box score (return vector?)
        #box_classes = tf.argmax(box_scores, axis=-1)

        # value of the highest box score (return vector?)
        #box_class_scores = tf.reduce_max(box_scores, axis=-1)

        box_classes = tf.argmax(box_scores, axis=-1)
        box_class_scores = tf.reduce_max(box_scores, axis=-1)

        prediction_mask = (box_class_scores >= threshold)

        boxes = tf.boolean_mask(boxes, prediction_mask)
        scores = tf.boolean_mask(box_class_scores, prediction_mask)
        classes = tf.boolean_mask(box_classes, prediction_mask)

        return self.non_max_suppression(boxes, scores, classes)

        # TODO: Problem is that tf.boolean_mask deletes the information about batch size and does not support keepdims.
        #  Tried to apply tf.map_fn, but it does not support inconsistent output shapes.
        #  Above I used typical for-loop but not sure if it won't be slowing down inference, refactor in future.
        # def filter_batch_box(boxes, box_class_scores, box_classes, prediction_mask):
        #     boxes = tf.boolean_mask(boxes, prediction_mask)
        #     scores = tf.boolean_mask(box_class_scores, prediction_mask)
        #     classes = tf.boolean_mask(box_classes, prediction_mask)
        # return tf.map_fn(lambda x: filter_batch_box(*x),
        #                  (boxes, box_class_scores, box_classes, prediction_mask),
        #                  dtype=(tf.float32, tf.float32, tf.int64),
        #                  infer_shape=False)

    def non_max_suppression(self, boxes, scores, classes, max_boxes=30, score_threshold=0.5, iou_threshold=0.5):
        """ Applying NMS, optimized box location for same classes"""

        max_boxes_tensor = tf.constant(max_boxes, dtype='int32')

        nms_index = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor,
                                                 iou_threshold=iou_threshold, score_threshold=score_threshold)
        boxes = tf.gather(boxes, nms_index)
        scores = tf.gather(scores, nms_index)
        classes = tf.gather(classes, nms_index)

        return boxes, scores, classes

    def convert_to_min_max_cord(self, boxes, anchors):

        box_xy = boxes[..., 0:2]
        box_wh = boxes[..., 2:4]

        # number of grids
        num_grids = tf.shape(boxes)[1]
        cell_size = tf.cast(tf.constant(self.config.image_width()) / num_grids, dtype=tf.float32)

        # create grid (B, 7, 7, A, 2) where we have the min absolute coordinates of each grid
        grid_i, grid_j = tf.meshgrid(tf.range(num_grids), tf.range(num_grids), indexing='ij')
        grid = tf.stack([grid_i, grid_j], axis=-1)
        grid = tf.expand_dims(grid, axis=2)
        grid = tf.tile(grid, [1, 1, len(anchors), 1])
        grid = tf.cast(grid, dtype=tf.float32)

        grid = tf.stack([grid] * self.config.batch_size())  # copy the grid to every batch

        # calculate relative coordinates, where xy is relative to cell_sizes and wh to appropriate anchor
        box_xy = box_xy + grid
        box_wh = (box_wh * np.array(anchors))

        # convert to absolute cordinates in px
        grid_length = cell_size * tf.cast(num_grids, dtype=tf.float32)
        box_xy = box_xy * cell_size
        box_wh = box_wh * grid_length

        # convert to min max coordinates
        box_mins = box_xy - (box_wh / 2.)
        box_maxes = box_xy + (box_wh / 2.)

        boxes = tf.concat([box_mins[..., 0:1],   # x_min
                           box_mins[..., 1:2],   # y_min
                           box_maxes[..., 0:1],  # x_max
                           box_maxes[..., 1:2],  # y_max
                           ], axis=-1)

        return boxes


if __name__ == '__main__':

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

    sess = tf.InteractiveSession()
    config_path = '/Users/adam.zvada/Documents/Dev/object-detection/config/yolo.yml'
    config = ConfigReader(config_path)
    network = CNNModel()
    yolo = YOLO(network, config)

    # a = [[0.5, 0.5, 64/448, 64/448], [0.0, 0.0, 64/448, 64/448]]
    # b = [[1.0, 1.0, 64/448, 64/448], [1.0, 1.0, 64/448, 64/448]]

    a = [[[0.5, 0.5, 100, 100], [0.5, 0.5, 150, 150], [0.5, 0.5, 10, 10]],
         [[0.5, 0.5, 100, 100], [0.5, 0.5, 150, 150], [0.5, 0.5, 10, 10]],
         [[0.5, 0.5, 100, 100], [0.5, 0.5, 150, 150], [0.5, 0.5, 10, 10]]]
    #a = [[0.5, 0.5, 1., 1.]]
    b = [[[0.5, 0.5, 100, 100], [0.5, 0.5, 150, 150], [0.5, 0.5, 10, 10]],
         [[0.5, 0.5, 100, 100], [0.5, 0.5, 150, 150], [0.5, 0.5, 10, 10]],
         [[0.5, 0.5, 100, 100], [0.5, 0.5, 150, 150], [0.5, 0.5, 10, 10]]]
    #b = [[0.5, 0.5, 1., 1.]]

    label_box = tf.convert_to_tensor(a, np.float32)
    pred_box = tf.convert_to_tensor(b, np.float32)

    # test IOU
    print(yolo.iou(label_box, pred_box, yolo.anchors[1]).eval())

    #print(yolo.convert_to_min_max_cord(label_box, pred_box).eval())

    # print(yolo.yolo_loss(predict=y_pred, label=y_true).eval())

    # ----------------CAN BE USED FOR TESTS----------------------------
    # ---------------TEST---coordinates conversion---------------------
    # from object_detection.dataset.udacity_object_dataset import UdacityObjectDataset
    # from object_detection.dataset.image_iterator import ImageIterator
    # import sys
    # import numpy
    # numpy.set_printoptions(threshold=sys.maxsize)
    #
    # sess = tf.InteractiveSession()
    #
    # config_path = '/Users/adam.zvada/Documents/Dev/object-detection/config/test.yml'
    # config = ConfigReader(config_path)
    #
    # network = CNNModel()
    # yolo = YOLO(network, config)
    #
    # dataset = UdacityObjectDataset(ConfigReader(config_path))
    # iterator = ImageIterator(sess, yolo, dataset, ConfigReader(config_path))
    #
    # input, test_handle = iterator.create_iterator(mode='train')
    #
    # label = sess.run(input['y'], feed_dict={iterator.handle_placeholder: test_handle})
    #
    # tf_label = tf.convert_to_tensor(label, np.float32)
    #
    # for g_x in range(0,7):
    #     for g_y in range(0, 7):
    #         for i, (rel_anchor_width, rel_anchor_height) in enumerate(yolo.anchors):
    #             cell_x = label[0][g_x][g_y][i][0]
    #             cell_y = label[0][g_x][g_y][i][1]
    #             a_w = label[0][g_x][g_y][i][2]
    #             a_h = label[0][g_x][g_y][i][3]
    #
    #             label[0][g_x][g_y][i][0] = int((((cell_x * 64) + g_x * 64) - (a_w * (rel_anchor_width * 448) / 2)))
    #             label[0][g_x][g_y][i][1] = int((((cell_y * 64) + g_y * 64) - (a_h * (rel_anchor_height * 448) / 2)))
    #
    #             label[0][g_x][g_y][i][2] = int((((cell_x * 64) + g_x * 64) + (a_w * (rel_anchor_width * 448) / 2)))
    #             label[0][g_x][g_y][i][3] = int((((cell_y * 64) + g_y * 64) + (a_h * (rel_anchor_height * 448) / 2)))
    #
    # print(label[0])
    #
    # dataset.df_true['x_min'] = dataset.df_true['x_min'] * 448/1920
    # dataset.df_true['y_min'] = dataset.df_true['y_min'] * 448/1200
    # dataset.df_true['x_max'] = dataset.df_true['x_max'] * 448/1920
    # dataset.df_true['y_max'] = dataset.df_true['y_max'] * 448/1200
    #
    # print(dataset.df_true[['x_min', 'y_min', 'x_max', 'y_max']])
    #
    # print(yolo.convert_to_min_max_cord(tf_label, 8).eval()[0])

    # ---------------TEST---coordinates conversion---------------------

