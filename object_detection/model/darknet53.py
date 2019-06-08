import tensorflow as tf
from object_detection.model.cnn_model import CNNModel
from object_detection.config.config_reader import ConfigReader


class DarkNet53(CNNModel):
    """
    DarkNet53 architecture for YOLOv3
    """
    def __init__(self, config: ConfigReader):
        super(DarkNet53, self).__init__()

        self.config = config

    def build_network(self, x, is_training):

        print(f'Input: {x.get_shape()}')

        conv = self.conv(x, filter_height=3, filter_width=3, num_filters=32,
                         stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv1')
        print(conv.get_shape())
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=64,
                         stride_x=2, stride_y=2, padding='SAME', training=is_training, scope_name='conv2')
        print(conv.get_shape())
        for i in range(1):
            conv = self.residual_conv_block(conv, num_filter_1=32, num_filter_2=64,
                                            kernel_1=(1, 1), kernel_2=(3, 3), name='residual_conv1')

        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=128,
                         stride_x=2, stride_y=2, padding='SAME', training=is_training, scope_name='conv5')

        for i in range(2):
            conv = self.residual_conv_block(conv, num_filter_1=64, num_filter_2=128,
                                            kernel_1=(1, 1), kernel_2=(3, 3), name='residual_conv2')

        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=256,
                         stride_x=2, stride_y=2, padding='SAME', training=is_training, scope_name='conv10')

        for i in range(8):
            conv = self.residual_conv_block(conv, num_filter_1=128, num_filter_2=256,
                                            kernel_1=(1, 1), kernel_2=(3, 3), name='residual_conv3')
        route_1 = conv

        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=512,
                         stride_x=2, stride_y=2, padding='SAME', training=is_training, scope_name='conv27')
        for i in range(8):
            conv = self.residual_conv_block(conv, num_filter_1=256, num_filter_2=512,
                                            kernel_1=(1, 1), kernel_2=(3, 3), name='residual_conv4')
        route_2 = conv

        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024,
                         stride_x=2, stride_y=2, padding='SAME', training=is_training, scope_name='conv36')

        for i in range(4):
            conv = self.residual_conv_block(conv, num_filter_1=512, num_filter_2=1024,
                                            kernel_1=(1, 1), kernel_2=(3, 3), name='residual_conv5')

        return self.pyramid_network(route_1, route_2, conv, is_training)

    def pyramid_network(self, route_1, route_2, x, is_training):

        # TODO: FIX INPUTS AND NAMING OUTPUTS (big, mid, small)
        conv = self.conv(x, filter_height=1, filter_width=1, num_filters=512, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=512, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=512, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')

        conv_big_obj = self.conv(conv, filter_height=3, filter_width=3, num_filters=1024, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')
        feature_map_1 = self.conv(conv_big_obj, filter_height=1, filter_width=1, num_filters=3*(5 + self.config.num_classes()),
                                  stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv1') # TODO: delete ACTIVATIONand BN

        conv = self.conv(x, filter_height=1, filter_width=1, num_filters=256, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')
        upsample = self.resize_conv(conv)

        conv = tf.concat([upsample, route_2], axis=-1, name='route_0')

        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=256, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=512, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=256, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=256, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=512, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')

        conv_mid_obj = self.conv(x, filter_height=3, filter_width=3, num_filters=256, stride_x=1, stride_y=1,
                                 padding='SAME', training=is_training, scope_name='conv1')
        feature_map_2 = self.conv(conv_mid_obj, filter_height=1, filter_width=1, num_filters=3 * (5 + self.config.num_classes()),
                                  stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv1') # TODO: delete ACTIVATIONand BN

        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=128, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')
        upsample = self.resize_conv(conv)

        conv = tf.concat([upsample, route_1], axis=-1, name='route_0')


        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=128, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=256, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=128, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')
        conv = self.conv(conv, filter_height=3, filter_width=3, num_filters=256, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')
        conv = self.conv(conv, filter_height=1, filter_width=1, num_filters=128, stride_x=1, stride_y=1,
                         padding='SAME', training=is_training, scope_name='conv1')

        conv_small_obj = self.conv(conv, filter_height=3, filter_width=3, num_filters=256, stride_x=1, stride_y=1,
                                 padding='SAME', training=is_training, scope_name='conv1')
        feature_map_3 = self.conv(conv_small_obj, filter_height=1, filter_width=1, num_filters=3 * (5 + self.config.num_classes()),
                                  stride_x=1, stride_y=1, padding='SAME', training=is_training, scope_name='conv1') # TODO: delete ACTIVATIONand BN

        return feature_map_1, feature_map_2, feature_map_3
        #return feature_map_3








