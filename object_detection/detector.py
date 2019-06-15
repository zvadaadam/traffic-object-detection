import tensorflow as tf
import numpy as np
from object_detection.config.config_reader import ConfigReader
from object_detection.model.darknet19 import DarkNet19
from object_detection.model.YOLO import YOLO
from object_detection.trainer.object_trainer import ObjectTrainer
from tensorflow.python.client import timeline


class Detector(object):

    def __init__(self, session: tf.Session(), config: ConfigReader, mode='predict'):
        self.config = config
        self.session = session
        self.mode = mode

        # TODO: load appropriate model type from config
        self.model = YOLO(DarkNet19(config), config)

        # init computational network graph
        self.model.build_model(mode=mode)

        # init tf.variables
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(init)

        # load model weight from config
        model_weight_path = self.config.restore_trained_model()
        self.model.init_saver(max_to_keep=2)
        self.model.load(self.session, model_weight_path)

    def train(self):
        pass

    def predict(self, image):

        if np.max(image) > 1.0:
            image /= 255

        # increased dim because of batch size 1
        image = np.expand_dims(image, axis=0)

        prediction = self.session.run(
            [self.model.detections], feed_dict={self.model.is_training: False, self.model.x: image}
        )

        return prediction[0]


if __name__ == '__main__':

    import datetime
    import numpy as np
    from object_detection.dataset.udacity_object_dataset import UdacityObjectDataset
    from object_detection.utils import image_utils
    from matplotlib.pyplot import imshow
    import matplotlib.pyplot as plt


    # config_path = '/home/zvadaada/traffic-object-detection/config/test.yml'
    config_path = '/Users/adam.zvada/Documents/Dev/object-detection/config/test.yml'
    config = ConfigReader(config_path)

    with tf.Session() as session:
        dataset = UdacityObjectDataset(config)
        test_df = dataset.test_dataset()
        images = np.array(test_df['image'].values.tolist()[0:90], dtype=np.float32)
        #label = np.array(test_df['label'].values.tolist()[0:1], dtype=np.float32)[0]

        detector = Detector(session, config=config)

        now = datetime.datetime.now()
        for image in images:
            output = detector.predict(image)

            img = image_utils.draw_boxes_PIL(image, output[0], output[1], output[2])
            imshow(img)
            plt.show()

        after = datetime.datetime.now()
        print(f'Inference Duration: {after - now}')


    # img = image[0] / 255
    # img = image_utils.draw_boxes_cv(img, output[0], output[1], output[2])
    # image_utils.plot_img(img)

    # img = image[0] / 255
    # for box in output[0]:
    #     img = image_utils.add_bb_to_img(img, box[0], box[1], box[2], box[3])
    # image_utils.plot_img(img)
