import tensorflow as tf
from object_detection.model.cnn_model import CNNModel
from object_detection.config.config_reader import ConfigReader


class YoloTiny(CNNModel):

    def __init__(self, config: ConfigReader):
        super(YoloTiny, self).__init__(config)

    def build_network(self, x, is_training):

        print('Input: {}'.format(x.get_shape()))

        # LAYER 1
        conv = self.conv(x, filter_height=7, filter_width=7, num_filters=64,
                         stride_x=2, stride_y=2, padding='SAME', training=is_training, scope_name='conv1')
        print('Conv1: {}'.format(conv.get_shape()))

        pool = self.max_pool(conv, filter_height=2, filter_width=2, stride_x=2, stride_y=2,
                             padding='VALID', scope_name='pool1')
        print('Pool1: {}'.format(pool.get_shape()))

        # LAYER 2
        conv = self.conv(pool, filter_height=3, filter_width=3, num_filters=192,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv2')
        print('Conv2: {}'.format(conv.get_shape()))

        pool = self.max_pool(conv, filter_height=2, filter_width=2, stride_x=2, stride_y=2,
                             padding='VALID', scope_name='pool1')
        print('Pool1: {}'.format(pool.get_shape()))

        # LAYER 3
        conv = self.conv(pool, filter_height=1, filter_width=1, num_filters=128,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv3')
        print('Conv3: {}'.format(conv.get_shape()))

        # LAYER 4
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=256,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv4')
        print('Conv4: {}'.format(conv.get_shape()))

        # LAYER 5
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=256,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv5')
        print('Conv5: {}'.format(conv.get_shape()))

        # LAYER 6
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv6')
        print('Conv6: {}'.format(conv.get_shape()))

        pool = self.max_pool(conv, filter_height=2, filter_width=2, stride_x=2, stride_y=2,
                             padding='VALID', scope_name='pool3')
        print('Pool3: {}'.format(pool.get_shape()))

        # LAYER 7
        conv = self.conv(pool, filter_height=1, filter_width=1, num_filters=256,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv7')
        print('Conv7: {}'.format(conv.get_shape()))

        # LAYER 8
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv8')
        print('Conv8: {}'.format(conv.get_shape()))

        # LAYER 9
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=256,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv9')
        print('Conv9: {}'.format(conv.get_shape()))

        # LAYER 10
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv10')
        print('Conv10: {}'.format(conv.get_shape()))

        # LAYER 11
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=256,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv11')
        print('Conv11: {}'.format(conv.get_shape()))

        # LAYER 12
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv12')
        print('Conv12: {}'.format(conv.get_shape()))

        # LAYER 13
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv13')
        print('Conv13: {}'.format(conv.get_shape()))

        # LAYER 14
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv14')
        print('Conv14: {}'.format(conv.get_shape()))

        # LAYER 15
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv15')
        print('Conv15: {}'.format(conv.get_shape()))

        # LAYER 16
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv16')
        print('Conv16: {}'.format(conv.get_shape()))

        pool = self.max_pool(conv, filter_height=2, filter_width=2, stride_x=2, stride_y=2,
                             padding='VALID', scope_name='pool4')
        print('Pool4: {}'.format(pool.get_shape()))

        # LAYER 17
        conv = self.conv(pool, filter_height=1, filter_width=1, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv17')
        print('Conv17: {}'.format(conv.get_shape()))

        # LAYER 18
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv18')
        print('Conv18: {}'.format(conv.get_shape()))

        # LAYER 19
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv19')
        print('Conv19: {}'.format(conv.get_shape()))

        # LAYER 20
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv20')
        print('Conv20: {}'.format(conv.get_shape()))

        # LAYER 21
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv21')
        print('Conv21: {}'.format(conv.get_shape()))

        # LAYER 22
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=2, stride_y=2, padding='SAME', training=is_training, scope_name='conv22')
        print('Conv22: {}'.format(conv.get_shape()))

        # LAYER 23
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv23')
        print('Conv23: {}'.format(conv.get_shape()))

        # LAYER 24
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv24')
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
        num_anchors = self.config.num_anchors()

        output_size = (grid_size * grid_size) * num_anchors * (5 + num_classes)

        fc2 = self.fully_connected(fc1, output_size, scope_name='fc7')
        print('FC2: {}'.format(fc2.get_shape()))

        logits = tf.reshape(fc2, shape=[self.config.batch_size(), grid_size, grid_size, num_anchors, 5 + num_classes])
        print('Logits: {}'.format(logits.get_shape()))

        return logits
