import tensorflow as tf
import numpy as np
from object_detection.model.base_model import ModelBase
from object_detection.config.config_reader import ConfigReader


class YOLO(ModelBase):

    def __init__(self, config: ConfigReader):
        super(YOLO, self).__init__(config)

        self.init_placeholders()

    def init_placeholders(self):

        image_height = self.config.image_height()
        image_width = self.config.image_width()
        num_classes = self.config.num_classes()
        grid_size = self.config.grid_size()
        boxes_per_cell = self.config.boxes_per_cell()

        self.x = tf.placeholder(tf.float32, [None, image_height, image_width, 3])
        self.y = tf.placeholder(tf.float32, [None, grid_size, grid_size, (boxes_per_cell*5) + num_classes])

    def build_model(self, input):

        x, y = input['x'], input['y']

        print('Building computation graph...')

        yolo_output = self.yolo(x)

        self.loss = self.yolo_loss(yolo_output, y)

        self.opt = self.optimizer(self.loss)

        self.acc = self.accuracy(self.logits, y)


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
        conv3 = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                          stride_x=1, stride_y=1, padding='SAME', scope_name='conv16')
        print('Conv16: {}'.format(conv3.get_shape()))

        pool = self.max_pool(conv, filter_height=2, filter_width=2, stride_x=2, stride_y=2,
                             padding='VALID', scope_name='pool4')
        print('Pool4: {}'.format(pool.get_shape()))

        # LAYER 17
        conv = self.conv(pool, filter_height=1, filter_width=1, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv17')
        print('Conv17: {}'.format(conv3.get_shape()))

        # LAYER 18
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv18')
        print('Conv18: {}'.format(conv3.get_shape()))

        # LAYER 19
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', scope_name='conv19')
        print('Conv19: {}'.format(conv3.get_shape()))

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
        cnn_output = tf.reshape(conv, [-1, feature_dim])
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

        return logits

    def yolo_loss(self, predict, label, lambda_cord=5, lambda_noobj=0.5):

        # INPUT: (?, grid_size, grid_size, bB*5 + num_classes)

        # TODO: we ASSUME B = 1

        loss_cord = self.cord_loss(label, predict, lambda_cord)
        loss_confidence = self.confidence_loss(label, predict, lambda_noobj)
        loss_class = self.classes_loss(label, predict)

        loss = tf.add(loss_cord, loss_confidence)
        loss = tf.add(loss, loss_class)

        return loss

    def optimizer(self, loss):
        return tf.train.AdamOptimizer().minimize(loss)

    def accuracy(self, logits, y):
        with tf.name_scope('loss'):
            acc = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))

        return acc

    def cord_loss(self, label, pred, lambda_cord=5):
        # INPUT: (?, grid_size, grid_size, bB*5 + num_classes)

        # TODO: we ASSUME B = 1

        # always is 0 or 1
        indicator_coord = tf.expand_dims(label[..., 4], axis=-1) # contains object(x>0) or does not contain object(x==0)
        #indicator_coord = tf.math.ceil(indicator_coord)

        # ---TESTING PURPOSE----
        # indicator_coord = tf.dtypes.cast(indicator_coord > 0.4, tf.float32)
        # print(indicator_coord.eval())

        with tf.name_scope('cord'):

            pred_cord = pred[..., 0:2]
            label_cord = label[..., 0:2]

            pred_size = pred[..., 2:4]
            label_size = label[..., 2:4]

            loss_cord = tf.multiply(indicator_coord, tf.square(pred_cord - label_cord))
            loss_cord = tf.reduce_sum(loss_cord)

            loss_size = tf.multiply(indicator_coord, tf.square(pred_size - label_size))
            loss_size = tf.reduce_sum(loss_size)

            loss = tf.scalar_mul(lambda_cord, tf.add(loss_size, loss_cord))

        return loss

    def confidence_loss(self, label, pred, lambda_noobj=0.5):

        indicator_noobj = (1 - label[..., 4])
        # ---TESTING PURPOSE----
        # indicator_noobj = tf.dtypes.cast(indicator_noobj > 0.6, tf.float32)
        # print(indicator_noobj.eval())
        # ---TESTING PURPOSE----

        indicator_obj = label[..., 4]
        # ---TESTING PURPOSE----
        # indicator_obj = tf.dtypes.cast(indicator_obj > 0.4, tf.float32)
        # print(indicator_obj.eval())
        # ---TESTING PURPOSE----

        with tf.name_scope('confidence'):

            confidence_label = label[..., 4]
            confidence_pred = pred[..., 4]

            loss_obj = tf.multiply(indicator_obj, tf.square(confidence_label - confidence_pred))
            loss_obj = tf.reduce_sum(loss_obj)

            loss_noobj = tf.multiply(indicator_noobj, tf.square(confidence_label - confidence_pred))
            loss_noobj = tf.reduce_sum(loss_noobj)
            loss_noobj = tf.scalar_mul(lambda_noobj, loss_noobj)

        return tf.add(loss_obj, loss_noobj)

    def classes_loss(self, label, pred):

        indicator_class = tf.expand_dims(label[..., 4], axis=-1)
        # ---TESTING PURPOSE----
        # indicator_class = tf.dtypes.cast(indicator_class > 0.6, tf.float32)
        # print(indicator_class.eval())
        # ---TESTING PURPOSE----

        with tf.name_scope('confidence'):
            class_label = label[..., 5:]
            class_pred = pred[..., 5:]
            loss_class = tf.multiply(indicator_class, tf.square(class_label - class_pred))
            loss_class = tf.reduce_sum(loss_class)

        return loss_class

    # def bb_intersection_over_union(self, boxA, boxB):
    #     # determine the (x, y)-coordinates of the intersection rectangle
    #     xA = max(boxA[0], boxB[0])
    #     yA = max(boxA[1], boxB[1])
    #     xB = min(boxA[2], boxB[2])
    #     yB = min(boxA[3], boxB[3])
    #
    #     # compute the area of intersection rectangle
    #     interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    #
    #     # compute the area of both the prediction and ground-truth
    #     # rectangles
    #     boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    #     boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    #
    #     # compute the intersection over union by taking the intersection
    #     # area and dividing it by the sum of prediction + ground-truth
    #     # areas - the interesection area
    #     iou = interArea / float(boxAArea + boxBArea - interArea)
    #
    #     # return the intersection over union value
    #     return iou



if __name__ == '__main__':

    sess = tf.InteractiveSession()

    num_classes = 10
    b = 1

    y_pred = tf.convert_to_tensor(np.random.rand(16, 13, 13, (b*5) + num_classes), np.float32)
    y_true = tf.convert_to_tensor(np.random.rand(16, 13, 13, (b*5) + num_classes), np.float32)

    # test = tf.expand_dims(y_true[..., 4], axis=-1)
    #
    # print(test.eval())
    #
    # test2 = y_true[..., 4]
    #
    # print(test2.eval())

    yolo = YOLO(sess, cofig=None)

    #print(yolo.cord_loss(label=y_true, pred=y_pred).eval())

    print(yolo.confidence_loss(label=y_true, pred=y_pred).eval())

    # yolo.loss(predict=y_pred, label=y_true)
