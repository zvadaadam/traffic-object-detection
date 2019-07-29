import sys
sys.path.append('/home/zvadaada/traffic-object-detection/')
import imutils
import tensorflow as tf
import numpy as np
from object_detection.config.config_reader import ConfigReader
from object_detection.model.darknet19 import DarkNet19
from object_detection.model.darknet53 import DarkNet53
from object_detection.model.YOLO import YOLO
from object_detection.trainer.object_trainer import ObjectTrainer
from tensorflow.python.client import timeline
from object_detection.dataset.all_dataset import AllDataset


class Detector(object):

    def __init__(self, session: tf.Session(), config: ConfigReader, mode='predict'):
        self.config = config
        self.session = session
        self.mode = mode

        # TODO: load appropriate model type from config
        #self.model = YOLO(DarkNet19(config), config)
        self.model = YOLO(DarkNet53(config), config)

        # # init computational network graph
        # self.model.build_model(mode=mode)
        #
        # # init tf.variables
        # init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # self.session.run(init)
        #
        # # load model weight from config
        # model_weight_path = self.config.restore_trained_model()
        # self.model.init_saver(max_to_keep=2)
        # self.model.load(self.session, model_weight_path)

    def init_prediction(self):
        # init computational network graph
        self.model.build_model(mode='predict')

        # init tf.variables
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(init)

        # load model weight from config
        model_weight_path = self.config.restore_trained_model()
        self.model.init_saver(max_to_keep=2)
        self.model.load(self.session, model_weight_path)


    def init_train_mode(self, model_path):
        # TODO: load appropriate model type from config
        self.model = YOLO(DarkNet19(config), config)

        # init computational network graph
        self.model.build_model(mode='train')

        # init tf.variables
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(init)

        # load model weight from config
        model_weight_path = self.config.restore_trained_model()
        self.model.init_saver(max_to_keep=2)
        self.model.load(self.session, model_weight_path)

    def train(self, dataset):

        # add additional options to trace the session execution
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        trainer = ObjectTrainer(self.session, self.model, dataset, self.config, options, run_metadata)
        trainer.train()

        # Create the Timeline object, and write it to a json file
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_01.json', 'w') as f:
            f.write(chrome_trace)

    def predict(self, image):

        if np.max(image) > 1.0:
            image = image/255

        # increased dim because of batch size 1
        image = np.expand_dims(image, axis=0)

        prediction = self.session.run(
            [self.model.detections], feed_dict={self.model.is_training: False,
                                                self.model.images: image}
        )

        return prediction[0]

if __name__ == '__main__':

    import os
    import datetime
    import cv2
    import numpy as np
    from object_detection.dataset.udacity_object_dataset import UdacityObjectDataset
    from object_detection.dataset.rovit_dataset import RovitDataset
    from object_detection.utils import image_utils
    from matplotlib.pyplot import imshow
    import matplotlib.pyplot as plt


    config_path = '/home/zvadaada/traffic-object-detection/config/yolo.yml'
    #config_path = '/Users/adam.zvada/Documents/Dev/object-detection/config/yolo.yml'
    #config_path = '/Users/adam.zvada/Documents/Dev/object-detection/config/test.yml'

    config = ConfigReader(config_path)

    # ------------TRAIN-----------------------
    with tf.Session() as session:
        dataset = AllDataset(config)
        dataset.load_dataset()

        print('DONE LOADING DATASET')

        detector = Detector(session, config=config)
        detector.train(dataset)
    # ------------TRAIN-----------------------


    # ------------PREDICTION-----------------------
    # with tf.Session() as session:
    #     #dataset = UdacityObjectDataset(config)
    #     dataset = RovitDataset(config)
    #     dataset.load_dataset()
    #     test_df = dataset.test_dataset()
    #
    #     image_filenames = test_df['image_filename'].values.tolist()[90:190]
    #
    #     image_filenames = ['/Users/adam.zvada/Documents/Dev/object-detection/dataset/0.png',
    #                        '/Users/adam.zvada/Documents/Dev/object-detection/dataset/1.png',
    #                        '/Users/adam.zvada/Documents/Dev/object-detection/dataset/2.png',
    #                        '/Users/adam.zvada/Documents/Dev/object-detection/dataset/3.png',
    #                        '/Users/adam.zvada/Documents/Dev/object-detection/dataset/4.png',
    #                        '/Users/adam.zvada/Documents/Dev/object-detection/dataset/5.png',
    #                        '/Users/adam.zvada/Documents/Dev/object-detection/dataset/6.png',
    #                        '/Users/adam.zvada/Documents/Dev/object-detection/dataset/7.png',
    #                        '/Users/adam.zvada/Documents/Dev/object-detection/dataset/8.png',
    #                        '/Users/adam.zvada/Documents/Dev/object-detection/dataset/9.png',
    #                        '/Users/adam.zvada/Documents/Dev/object-detection/dataset/12.png',
    #                        '/Users/adam.zvada/Documents/Dev/object-detection/dataset/11.png',
    #                        '/Users/adam.zvada/Documents/Dev/object-detection/dataset/13.jpg',
    #                        '/Users/adam.zvada/Documents/Dev/object-detection/dataset/14.jpg']
    #
    #     images = []
    #     for image_filename in image_filenames:
    #         #image = cv2.imread(os.path.join(config.udacity_dataset_path(), image_filename))
    #         image = cv2.imread(image_filename)
    #
    #         resized_img = cv2.resize(image, (config.image_width(), config.image_height()),
    #                                 interpolation=cv2.INTER_NEAREST)
    #
    #         # resized_img = imutils.resize(image, width=config.image_width(), height=config.image_height())
    #
    #         resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)  # changed colors
    #         print(resized_img.shape)
    #
    #         # imshow(resized_img)
    #         # plt.show()
    #
    #         images.append(resized_img)
    #
    #     image = np.array(images)
    #
    #     #label = np.array(test_df['label'].values.tolist()[0:1], dtype=np.float32)[0]
    #
    #     detector = Detector(session, config=config)
    #     detector.init_prediction()
    #
    #     now = datetime.datetime.now()
    #     for image in images:
    #         output = detector.predict(image)
    #
    #         img = image_utils.draw_boxes_PIL(image, output[0], output[1], output[2])
    #         imshow(img)
    #         plt.show()
    #
    #     after = datetime.datetime.now()
    #     print(f'Inference Duration: {after - now}')
    # ------------PREDICTION-----------------------
