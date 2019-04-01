import tensorflow as tf
from object_detection.model.base_model import ModelBase
from object_detection.config.config_reader import ConfigReader


class CNNModel(ModelBase):

    def __init__(self, config: ConfigReader):
        super(CNNModel, self).__init__(config)

        self.loss = None
        self.acc = None
        self.opt = None
        self.logits = None
        self.x = None
        self.y = None


    def normalization(self, x, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta,
                                                  bias=bias, name=name)

    def conv(self, inputs, filter_height, filter_width, num_filters,
             stride_x, stride_y, padding, scope_name='conv'):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            # input_channels = inputs.shape[-1]
            #
            # weights = tf.get_variable('weights', [filter_height, filter_width,
            #                                       input_channels, num_filters],
            #                           initializer=tf.truncated_normal_initializer())
            #
            # biases = tf.get_variable('biases', [num_filters], initializer=tf.random_normal_initializer())
            #
            # conv = tf.nn.conv2d(inputs, weights, strides=[1, stride_y, stride_x, 1],
            #                     padding=padding)

            x = tf.layers.conv2d(inputs, kernel_size=(filter_height, filter_width), filters=num_filters,
                                 strides=(stride_x, stride_y), padding=padding,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 bias_initializer=tf.zeros_initializer())
            x = tf.layers.batch_normalization(x, training=True, momentum=0.99, epsilon=0.001, center=True,
                                              scale=True)

        return tf.nn.leaky_relu(x, name=scope.name, alpha=0.1)

    def max_pool(self, inputs, filter_height, filter_width,
                 stride_x, stride_y, padding='VALID', scope_name='pool'):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            pool = tf.nn.max_pool(inputs, ksize=[1, filter_height, filter_width, 1],
                                  strides=[1, stride_y, stride_x, 1],
                                  padding=padding)
        return pool

    def fully_connected(self, inputs, num_outputs, scope_name='fully_conncted'):

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            input_dim = inputs.shape[-1]

            w = tf.get_variable('weights', [input_dim, num_outputs],
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('biases', [num_outputs],
                                initializer=tf.constant_initializer(0.0))
            logit = tf.matmul(inputs, w) + b

        return logit
