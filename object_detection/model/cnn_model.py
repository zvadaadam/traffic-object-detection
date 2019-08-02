import tensorflow as tf
from object_detection.model.base_model import ModelBase
from object_detection.config.config_reader import ConfigReader


class CNNModel(object):

    def __init__(self):
        pass

    def build_network(self, x, is_training):
        raise NotImplementedError

    def residual_conv_block(self, input, num_filter_1, num_filter_2, kernel_1, kernel_2, name):

        with tf.variable_scope(name):
            output = tf.keras.layers.Conv2D(filters=num_filter_1, kernel_size=kernel_1, padding='SAME')(input)
            #output = tf.keras.activations.relu(output)
            output = tf.keras.layers.LeakyReLU(alpha=0.1)(output)
            print(f'InResConv1: {output.get_shape()}')

            output = tf.keras.layers.Conv2D(filters=num_filter_2, kernel_size=kernel_2, padding='SAME')(output)
            #output = tf.keras.activations.relu(output)
            output = tf.keras.layers.LeakyReLU(alpha=0.1)(output)
            print(f'InResConv2: {output.get_shape()}')

            residual_output = input + output

            #residual_output = tf.keras.activations.relu(residual_output)
            residual_output = tf.keras.activations.linear(residual_output)

        return residual_output


    def normalization(self, x, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta,
                                                  bias=bias, name=name)

    def conv(self, inputs, filter_height, filter_width, num_filters,
             stride_x, stride_y, padding, training, activate=True, bn=True, scope_name='conv'):

        with tf.variable_scope(scope_name) as scope:

            # darkent uses top left padding
            if stride_x > 1:
                inputs = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
                padding = 'VALID'

            output = tf.layers.Conv2D(kernel_size=(filter_height, filter_width), filters=num_filters,
                                 strides=(stride_x, stride_y), padding=padding, use_bias=False,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 bias_initializer=tf.zeros_initializer(), activation=None)(inputs)

            # output = tf.contrib.layers.batch_norm(output, center=True, scale=True,
            #                                       is_training=training, renorm=True, scope='batch_norm')

            # output = tf.layers.batch_normalization(output, beta_initializer=tf.zeros_initializer(),
            #                                        gamma_initializer=tf.ones_initializer(),
            #                                        moving_mean_initializer=tf.zeros_initializer(),
            #                                        moving_variance_initializer=tf.ones_initializer(),
            #                                        training=True)

            # output = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(filter_height, filter_width),
            #                                 strides=(stride_x, stride_y), padding=padding)(inputs)

            #output = tf.layers.batch_normalization(output, epsilon=0.001, scale=True, training=training, momentum=0.9)
            # batch_normed = tf.keras.layers.BatchNormalization(momentum=0.9, renorm=True)(output, training=True)

            #output = tf.keras.activations.relu(batch_normed)
            #output = tf.nn.leaky_relu(batch_normed, name=scope.name, alpha=0.1)
            #output = tf.keras.layers.LeakyReLU(alpha=0.1)(batch_normed)

            if bn:
                output = tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=0.9, renorm=True)(output, training=training)
            if activate:
                output = tf.keras.layers.LeakyReLU(alpha=0.1)(output)

        print(f'Conv: {output.get_shape()}')

        return output

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

    def deconv(self, inputs):

        num_filter = input.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(inputs, num_filter, kernel_size=2, padding='same',
                                            strides=(2, 2), kernel_initializer=tf.random_normal_initializer())

        return output

    def resize_conv(self, inputs):
        input_shape = tf.shape(inputs)
        output = tf.image.resize_nearest_neighbor(inputs, (input_shape[1] * 2, input_shape[2] * 2))

        return output


