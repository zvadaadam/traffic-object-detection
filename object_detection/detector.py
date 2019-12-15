import sys
sys.path.append('/home/zvadaada/traffic-object-detection/')
import imutils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from object_detection.utils import image_utils
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

    def init_eval(self):
        # init computational network graph
        self.model.build_model(mode='eval')

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

    # def eval(self, dataset):
    #
    #     # add additional options to trace the session execution
    #     options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #     run_metadata = tf.RunMetadata()
    #
    #     trainer = ObjectTrainer(self.session, self.model, dataset, self.config, options, run_metadata)
    #     trainer.eval(num_iterations=10)
    #
    #     # Create the Timeline object, and write it to a json file
    #     fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    #     chrome_trace = fetched_timeline.generate_chrome_trace_format()
    #     with open('timeline_01.json', 'w') as f:
    #         f.write(chrome_trace)

    def eval(self, image, label):

        loss_small, loss_medium, loss_large, loss_class, loss_confidence, loss_noobj, loss_obj, \
        loss_size, loss_cord, detections = self.session.run(
            [self.model.loss_small,
             self.model.loss_medium,
             self.model.loss_large,
             self.model.loss_class,
             self.model.loss_confidence,
             self.model.loss_noobj,
             self.model.loss_obj,
             self.model.loss_size,
             self.model.loss_cord,
             self.model.detections], feed_dict={self.model.is_training: True,
                                                self.model.images: image,
                                                self.model.y_small: label[0],
                                                self.model.y_medium: label[1],
                                                self.model.y_large: label[2],
                                                }
        )

        print(loss_small)
        print(loss_medium)
        print(loss_large)
        print(f'Class: {loss_class}, conf: {loss_confidence}, noobj: {loss_noobj}, obj: {loss_obj}, size: {loss_size}, cord: {loss_cord}')

        img = image_utils.draw_boxes_PIL(image[0], detections[0], detections[1], detections[2])
        plt.imshow(img)
        plt.show()

        label_to_boxes = self.model.label_to_boxes()
        transformed_labels = self.session.run(label_to_boxes, feed_dict={self.model.y_small: label[0],
                                                                    self.model.y_medium: label[1],
                                                                    self.model.y_large: label[2]})

        img = image_utils.draw_boxes_PIL(img, transformed_labels[0], transformed_labels[1], transformed_labels[2])
        plt.imshow(img)
        plt.show()


    def predict(self, image):

        if np.max(image) > 1.0:
            image = image/255

        # increased dim because of batch size 1
        image = np.expand_dims(image, axis=0)

        prediction = self.session.run(
            [self.model.detections], feed_dict={self.model.is_training: True,
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
    from object_detection.dataset.image_iterator import ImageIterator
    from object_detection.utils import image_utils
    import matplotlib.pyplot as plt
    from tensorflow.python import debug as tf_debug

    #config_path = '/home/zvadaada/traffic-object-detection/config/yolo.yml'
    config_path = '/Users/adam.zvada/Documents/Dev/object-detection/config/yolo.yml'
    #config_path = '/Users/adam.zvada/Documents/Dev/object-detection/config/test.yml'

    config = ConfigReader(config_path)

    # ------------TRAIN-----------------------
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # with tf.Session(config=tf_config) as session:
    #
    #     # session = tf_debug.TensorBoardDebugWrapperSession(session, 'PRGA-004810.local:6064', send_traceback_and_source_code=False)
    #
    #     dataset = AllDataset(config)
    #     dataset.load_dataset()
    #
    #     print('DONE LOADING DATASET')
    #
    #     # import io
    #     # for filename in np.asarray(dataset.train_df['image_filename'].values.tolist()):
    #     #     try:
    #     #         io.imread(filename)
    #     #     except:
    #     #         print(filename)
    #
    #     detector = Detector(session, config=config)
    #     detector.train(dataset)
    # ------------TRAIN-----------------------


    # ------------EVAL-----------------------
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # with tf.Session(config=tf_config) as session:
    #
    #     # session = tf_debug.TensorBoardDebugWrapperSession(session, 'PRGA-004810.local:6064', send_traceback_and_source_code=False)
    #
    #     dataset = AllDataset(config)
    #     dataset.load_dataset()
    #
    #     print('DONE LOADING DATASET')
    #
    #     # import io
    #     # for filename in np.asarray(dataset.train_df['image_filename'].values.tolist()):
    #     #     try:
    #     #         io.imread(filename)
    #     #     except:
    #     #         print(filename)
    #
    #     detector = Detector(session, config=config)
    #     detector.eval(dataset)
    # ------------TRAIN-----------------------


    # ------------PREDICTION-----------------------
    with tf.Session() as session:
        #dataset = AllDataset(config)
        # dataset = UdacityObjectDataset(config)
        # dataset = RovitDataset(config)
        #dataset.load_dataset()
        #test_df = dataset.test_dataset()
        #
        # image_filenames = test_df['image_filename'].values.tolist()[90:190]

        image_filenames = [
                           '/Users/adam.zvada/Documents/Dev/object-detection/dataset/fuzee-traffic-dataset/JPEGImages/DJI_0413.MP4#t=0.2.jpg',
                           '/Users/adam.zvada/Documents/Dev/object-detection/dataset/0.png',
                           '/Users/adam.zvada/Documents/Dev/object-detection/dataset/1.png',
                           '/Users/adam.zvada/Documents/Dev/object-detection/dataset/2.png',
                           '/Users/adam.zvada/Documents/Dev/object-detection/dataset/3.png',
                           '/Users/adam.zvada/Documents/Dev/object-detection/dataset/4.png',
                           '/Users/adam.zvada/Documents/Dev/object-detection/dataset/5.png',
                           '/Users/adam.zvada/Documents/Dev/object-detection/dataset/6.png',
                           '/Users/adam.zvada/Documents/Dev/object-detection/dataset/7.png',
                           '/Users/adam.zvada/Documents/Dev/object-detection/dataset/8.png',
                           '/Users/adam.zvada/Documents/Dev/object-detection/dataset/9.png',
                           '/Users/adam.zvada/Documents/Dev/object-detection/dataset/12.png',
                           '/Users/adam.zvada/Documents/Dev/object-detection/dataset/11.png',
                           '/Users/adam.zvada/Documents/Dev/object-detection/dataset/13.jpg',
                           '/Users/adam.zvada/Documents/Dev/object-detection/dataset/14.jpg']

        images = []
        for image_filename in image_filenames:

            #image = cv2.imread(os.path.join(config.udacity_dataset_path(), image_filename))
            image = cv2.imread(image_filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # changed colors

            resized_img = image_utils.letterbox_image_2(image, (config.image_width(), config.image_height()))

            images.append(resized_img)

            plt.imshow(resized_img)
            plt.show()


        detector = Detector(session, config=config)
        detector.init_prediction()

        now = datetime.datetime.now()
        for image in images:

            print("NEW PREDICTION...")

            output = detector.predict(image)
            print(output)

            img = image_utils.draw_boxes_PIL(image, output[0], output[1], output[2])
            plt.imshow(img)
            plt.show()

        after = datetime.datetime.now()
        print(f'Inference Duration: {after - now}')
    # ------------PREDICTION-----------------------

    # ------------PREDICTION-USING-TF.DATASET------
    # with tf.Session() as session:
    #     dataset = AllDataset(config)
    #     dataset.load_dataset()
    #
    #     detector = Detector(session, config=config)
    #     detector.init_prediction()
    #     iterator = ImageIterator(session, dataset, config, detector.model)
    #     x, y, handler = iterator.create_iterator(mode='test')
    #
    #     images, labels = session.run((x, y))
    #
    #     now = datetime.datetime.now()
    #     for image in images:
    #         print("NEW PREDICTION...")
    #
    #         output = detector.predict(image)
    #         print(output)
    #
    #         img = image_utils.draw_boxes_PIL(image, output[0], output[1], output[2])
    #         plt.imshow(img)
    #         plt.show()
    #
    #     after = datetime.datetime.now()
    #     print(f'Inference Duration: {after - now}')
    # ------------PREDICTION-USING-TF.DATASET------
