import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from object_detection.config.config_reader import ConfigReader
from object_detection.dataset.dataset_base import DatasetBase
from object_detection.model.base_model import ModelBase


class ImageIterator(object):

    def __init__(self, session: tf.Session, dataset: DatasetBase, config: ConfigReader, model=None):
        self.session = session
        self.model = model
        self.dataset = dataset
        self.config = config

        self.handle_placeholder = tf.placeholder(tf.string, shape=[])

    def create_iterator_from_tfrecords(self, tfrecords_path=None, mode='train'):

        if tfrecords_path == None:
            if mode == 'train':
                tfrecords_path = self.config.tfrecords_train_path()
            elif mode == 'test':
                tfrecords_path = self.config.tfrecords_test_path()
            else:
                raise Exception(f'Mode {mode} does not exist.')

        dataset = tf.data.TFRecordDataset([tfrecords_path])
        dataset = dataset.map(self.preprocess_tfrecords, num_parallel_calls=8)
        dataset = dataset.cache()
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=188))  # TODO: num images
        # dataset = dataset.shuffle(self.config.batch_size()*2)
        dataset = dataset.batch(self.config.batch_size(), drop_remainder=True)
        #dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(buffer_size=2)

        dataset_iterator = dataset.make_initializable_iterator()

        dataset_handle = self.session.run(dataset_iterator.string_handle())

        x, y = dataset_iterator.get_next()

        self.session.run(dataset_iterator.initializer)

        # show data pipeline performance
        self.time_pipeline(dataset_iterator)

        return x, y, dataset_handle

    def preprocess_tfrecords(self, data_tfrecord):

        assert self.config.image_width() == self.config.image_height()
        image_size = self.config.image_width()
        num_cells = self.config.grid_size()
        num_anchors = self.config.num_anchors()
        num_classes = self.config.num_classes()

        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string)
        }

        # Extract the data record
        sample = tf.parse_single_example(data_tfrecord, features)

        image = tf.image.decode_image(sample['image'], channels=3)
        image = tf.reshape(image, [image_size, image_size, 3])
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        labels = tf.decode_raw(sample['labels'], out_type=tf.float32)
        labels = tf.cast(tf.reshape(labels, shape=[num_cells, num_cells, num_anchors, 5 + num_classes]), dtype=tf.float32)

        return image, labels

    def create_iterator(self, mode='train'):

        if mode == 'train':
            df = self.dataset.train_dataset()
        else:
            df = self.dataset.test_dataset()

        dataset = tf.data.Dataset.from_tensor_slices(
            (self.model.image_paths, self.model.y_small, self.model.y_medium, self.model.y_large)
        )

        num_images = len(df)
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=num_images))
        dataset = dataset.map(self.preprocess_record, num_parallel_calls=8)
        dataset = dataset.map(self.image_augmentation, num_parallel_calls=8)
        dataset = dataset.batch(self.config.batch_size(), drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        dataset_iterator = dataset.make_initializable_iterator()

        dataset_handle = self.session.run(dataset_iterator.string_handle())

        x, y_small, y_medium, y_large = dataset_iterator.get_next()
        y = [y_small, y_medium, y_large]

        feed = {
            self.model.image_paths: np.asarray(df['image_filename'].values.tolist()),
            self.model.y_small: np.asarray(df['label_small'].values.tolist(), dtype=np.float32),
            self.model.y_medium: np.asarray(df['label_medium'].values.tolist(), dtype=np.float32),
            self.model.y_large: np.asarray(df['label_large'].values.tolist(), dtype=np.float32),
        }

        self.session.run(dataset_iterator.initializer, feed_dict=feed)

        # show data pipeline performance
        # self.time_pipeline(dataset_iterator)

        return x, y, dataset_handle

    # TODO: refactor class to not use static
    @staticmethod
    def predict_iterator(x, y):

        def normalization(image, label):
            #image = tf.cond(tf.math.reduce_max(image) > 1.0, lambda: image / 255, lambda: image)

            image = image / 255

            return image, label

        num_images = len(x)

        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.map(normalization)
        dataset = dataset.batch(num_images)

        dataset = dataset.make_one_shot_iterator()

        input_x, input_y = dataset.get_next()

        inputs = {
            'x': input_x,
            'y': input_y,
        }

        return inputs

    def time_pipeline(self, dataset_iterator):

        print('Running pipeline test...')

        import time
        overall_start = time.time()
        self.session.run(dataset_iterator.get_next())

        start = time.time()
        for i in tqdm(range(0, self.config.num_iterations())):
            self.session.run(dataset_iterator.get_next())

            # np.set_printoptions(formatter={'float_kind': '{:f}'.format})
            # print(f'Image: {input[0]}')
            # print(f'Lables: {input[1]}')

        end = time.time()

        duration = end - start
        print('Data Pipeline Performance:')
        print(" {} batches: {} s".format(self.config.num_iterations(), duration))
        print(" {:0.5f} Images/s".format(self.config.batch_size() * self.config.num_iterations() / duration))
        print(" Total time: {}s".format(end - overall_start))

    def preprocess_record(self, image, label_small, label_medium, label_large):

        # udacity_image_path = os.path.join(self.config.udacity_dataset_path(), image)
        # rovit_image_path = os.path.join(self.config.rovit_dataset_path(), 'JPEGImages', image)
        #
        # image_path = ''
        # if os.path.exists(udacity_image_path):
        #     image_path = udacity_image_path
        # elif os.path.exists(rovit_image_path):
        #     image_path = rovit_image_path
        # else:
        #     raise Exception(f'Image {image} not found...')

        # load image
        img_raw = tf.read_file(image)
        #img_raw = tf.Print(img_raw, [image])
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # image normalization
        img = tf.image.resize_images(img, [self.config.image_width(), self.config.image_height()],
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # image normalization
        #img = tf.cond(tf.math.reduce_max(img) > 1.0, lambda: img/255, lambda: img)

        return img, label_small, label_medium, label_large

    def image_augmentation(self, image, label_small, label_medium, label_large):
        #image = tf.image.random_flip_left_right(image)

        image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

        # Make sure the image is still in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label_small, label_medium, label_large

if __name__ == '__main__':

    config = ConfigReader()

    import matplotlib.pyplot as plt
    import cv2
    from object_detection.model.darknet19 import DarkNet19
    from object_detection.model.darknet53 import DarkNet53
    from object_detection.model.YOLO import YOLO
    from object_detection.utils import image_utils
    #model = YOLO(DarkNet19(config), config)
    model = YOLO(DarkNet53(config), config)

    from object_detection.dataset.rovit_dataset import RovitDataset
    from object_detection.dataset.bdd_dataset import BddDataset
    dataset = BddDataset(config)
    dataset.load_dataset()

    for _ in range(0, 1):
        with tf.Session() as session:
            iterator = ImageIterator(session, dataset, config, model)
            # iterator.create_iterator_from_tfrecords(mode='train')
            x, y, handler = iterator.create_iterator(mode='train')

            images, labels = session.run((x, y))

            test_iou = model.test_iou()
            output = session.run(test_iou, feed_dict={model.test_large: labels[2], model.y_large: labels[2]})
            print(output)

            # check bb conversion
            # label_to_boxes = model.label_to_boxes()
            # transformed_labels = session.run(label_to_boxes, feed_dict={model.y_medium: labels[1]})
            #
            # image = image_utils.draw_boxes_PIL(images[0], boxes=transformed_labels[0], scores=transformed_labels[1],
            #                                    classes=transformed_labels[2])
            # plt.imshow(image)
            # plt.show()

        # for image in images:
            #     image = image_utils.draw_boxes_PIL(image, boxes=transformed_labels[0], scores=transformed_labels[1],
            #                                        classes=transformed_labels[2])
            #     plt.imshow(image)
            #     plt.show()

