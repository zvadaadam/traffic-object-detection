import tensorflow as tf
from object_detection.model.base_model import ModelBase


class TinyYOLO(ModelBase):

    def __init__(self, cofig):

        super(ModelBase, self).__init__(cofig)

    def build_model(self, x, y):
        pass

    def tiny_yolo(self, input):

        # layer 1
        conv1 = self.conv(input, filter_height=3, filter_width=3, num_filters=16,
                  stride_x=1, stride_y=1, padding='SAME', scope_name='conv1')
        print('Conv1: {}'.format(conv1.get_shape()))

        pool1 = self.max_pool(conv1, filter_height=2, filter_width=2, stride_x=2, stride_y=2,
                              padding='VALID', scope_name='pool1')
        print('Pool1: {}'.format(pool1.get_shape()))

        # layer 2


    def normalization(self, x, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta,
                                                  bias=bias, name=name)

    def conv(self, inputs, filter_height, filter_width, num_filters,
             stride_x, stride_y, padding, scope_name='conv'):

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            input_channels = inputs.shape[-1]

            weights = tf.get_variable('weights', [filter_height, filter_width,
                                                  input_channels, num_filters],
                                      initializer=tf.truncated_normal_initializer())

            biases = tf.get_variable('biases', [num_filters], initializer=tf.random_normal_initializer())

            conv = tf.nn.conv2d(inputs, weights, strides=[1, stride_y, stride_x, 1],
                                padding=padding)

        return tf.nn.relu(conv + biases, name=scope.name)

    def max_pool(self, inputs, filter_height, filter_width,
                 stride_x, stride_y, padding='VALID', scope_name='pool'):

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            pool = tf.nn.max_pool(inputs, ksize=[1, filter_height, filter_width, 1],
                                  strides=[1, stride_y, stride_x, 1],
                                  padding=padding)
        return pool



