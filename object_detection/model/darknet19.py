import tensorflow as tf
from object_detection.model.cnn_model import CNNModel
from object_detection.config.config_reader import ConfigReader


class DarkNet19(CNNModel):
    """
    DarkNet19 architecture for YOLOv2
    """
    def __init__(self, config: ConfigReader):
        super(DarkNet19, self).__init__()

        self.config = config

    def build_network(self, x, is_training):

        print(f'Input: {x.get_shape()}')

        conv = self.conv(x, filter_height=3, filter_width=3, num_filters=32,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv1')

        pool = self.max_pool(conv, filter_height=2, filter_width=2, stride_x=2, stride_y=2,
                             padding='VALID', scope_name='pool1')

        conv = self.conv(pool, filter_height=3, filter_width=3, num_filters=64,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv2')

        pool = self.max_pool(conv, filter_height=2, filter_width=2, stride_x=2, stride_y=2,
                             padding='VALID', scope_name='pool2')

        conv = self.conv(pool, filter_height=3, filter_width=3, num_filters=128,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv3')

        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=64,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv4')

        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=128,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv5')

        pool = self.max_pool(conv, filter_height=2, filter_width=2, stride_x=2, stride_y=2,
                             padding='VALID', scope_name='pool3')

        conv = self.conv(pool, filter_height=3, filter_width=3, num_filters=256,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv6')

        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=128,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv7')

        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=256,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv8')

        pool = self.max_pool(conv, filter_height=2, filter_width=2, stride_x=2, stride_y=2,
                             padding='VALID', scope_name='pool4')

        conv = self.conv(pool, filter_height=3, filter_width=3, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv9')

        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=256,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv10')

        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv11')

        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=256,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv12')

        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv13')

        pool = self.max_pool(conv, filter_height=2, filter_width=2, stride_x=2, stride_y=2,
                             padding='VALID', scope_name='pool5')

        conv = self.conv(pool, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv14')

        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv15')

        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv16')

        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=512,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv17')

        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv18')

        conv = self.conv(conv, filter_height=1, filter_width=1,
                         num_filters=self.config.num_anchors() * (5 + self.config.num_classes()),
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv19')

        # output = tf.keras.layers.GlobalAveragePooling2D()(conv)
        # output = tf.keras.layers.Dense(1000, activation='softmax')(output)

        # TODO: reshape to ambigous batch size
        logits = tf.reshape(conv, shape=[self.config.batch_size(),
                                         self.config.grid_size(), self.config.grid_size(),
                                         self.config.boxes_per_cell(),
                                         (5 + self.config.num_classes())])

        print(f'Logit: {logits.get_shape()}')

        return logits

