import tensorflow as tf
from object_detection.config.config_reader import ConfigReader
from object_detection.model.darknet19 import DarkNet19
from object_detection.model.YOLO import YOLO
from object_detection.dataset.image_iterator import ImageIterator


class Detector(object):

    def __init__(self, config: ConfigReader):
        self.config = config

        # TODO: load appropriate model type from config
        darknet = DarkNet19(config)
        self.model = YOLO(darknet, config)

    def train(self):
        pass

    def predict(self, img, tmp_label):

        model_weight_path = self.config.restore_trained_model()

        with tf.Session() as session:
            # TODO: delete tmp_label
            input = ImageIterator.predict_iterator(img, tmp_label)

            # TODO: refactor build_model for prediction
            self.model.build_model(input)

            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            session.run(init)

            self.model.init_saver(max_to_keep=2)
            self.model.load(session, model_weight_path)

            input, output, loss = session.run(
                [self.model.input, self.model.eval, self.model.get_loss()],
                feed_dict={self.model.is_training: False})

        return output, loss


if __name__ == '__main__':

    import datetime
    import numpy as np
    from object_detection.dataset.udacity_object_dataset import UdacityObjectDataset
    from object_detection.utils import image_utils

    # config_path = '/home/zvadaada/traffic-object-detection/config/test.yml'
    config_path = '/Users/adam.zvada/Documents/Dev/object-detection/config/test.yml'
    config = ConfigReader(config_path)

    detector = Detector(config=config)

    dataset = UdacityObjectDataset(config)
    test_df = dataset.test_dataset()

    image = np.array(test_df['image'].values.tolist()[0:1], dtype=np.float32)
    label = np.array(test_df['label'].values.tolist()[0:1], dtype=np.float32)

    now = datetime.datetime.now()
    output, loss = detector.predict(image, label)
    after = datetime.datetime.now()
    print(f'Inference Duration: {after - now}')
    print(f'Loss: {loss}')

    # img = image[0] / 255
    # img = image_utils.draw_boxes_cv(img, output[0], output[1], output[2])
    # image_utils.plot_img(img)

    # img = image[0] / 255
    # for box in output[0]:
    #     img = image_utils.add_bb_to_img(img, box[0], box[1], box[2], box[3])
    # image_utils.plot_img(img)
    #

    img = image[0] / 255
    img = image_utils.draw_boxes_PIL(img, output[0], output[1], output[2])
    from matplotlib.pyplot import imshow
    import matplotlib.pyplot as plt
    imshow(img)
    plt.show()