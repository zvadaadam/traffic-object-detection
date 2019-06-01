import tensorflow as tf
import numpy as np
import pandas as pd
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
        inputs = {
            'x': x,
            'y': y,
        }

        self.session.run(dataset_iterator.initializer)

        # show data pipeline performance
        self.time_pipeline(dataset_iterator)

        return inputs, dataset_handle

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

        dataset = tf.data.Dataset.from_tensor_slices((self.model.x, self.model.y))

        num_images = len(self.dataset.train_dataset())
        dataset = dataset.map(self.normalization)
        dataset = dataset.cache()
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=num_images))
        dataset = dataset.batch(self.config.batch_size(), drop_remainder=True)
        #dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(buffer_size=1)

        dataset_iterator = dataset.make_initializable_iterator()

        dataset_handle = self.session.run(dataset_iterator.string_handle())

        x, y = dataset_iterator.get_next()
        inputs = {
            'x': x,
            'y': y,
        }

        if mode == 'train':
            df = self.dataset.train_dataset()
        else:
            df = self.dataset.test_dataset()

        images = np.array(df['image'].values.tolist(), dtype=np.float32)
        labels = np.array(df['label'].values.tolist())

        for i in range(0, len(images)):
            img = images[i]
            img = img.astype('float32')

            if img.max() > 1.0:
                img /= 255.0

        feed = {
            self.model.x: images,
            self.model.y: labels
        }

        self.session.run(dataset_iterator.initializer, feed_dict=feed)

        # show data pipeline performance
        self.time_pipeline(dataset_iterator)

        return inputs, dataset_handle

    def time_pipeline(self, dataset_iterator):

        import time
        overall_start = time.time()
        self.session.run(dataset_iterator.get_next())

        start = time.time()
        for i in range(0, self.config.num_iterations()):
            input = self.session.run(dataset_iterator.get_next())

            np.set_printoptions(formatter={'float_kind': '{:f}'.format})
            print(f'Image: {input[0]}')
            print(f'Lables: {input[1]}')

        end = time.time()

        duration = end - start
        print('Data Pipeline Performance:')
        print(" {} batches: {} s".format(self.config.num_iterations(), duration))
        print(" {:0.5f} Images/s".format(self.config.batch_size() * self.config.num_iterations() / duration))
        print(" Total time: {}s".format(end - overall_start))

    def normalization(self, image, label):

        image = tf.cond(tf.math.reduce_max(image) > 1.0, lambda: image/255, lambda: image)

        return image, label

if __name__ == '__main__':

    config = ConfigReader()

    from object_detection.dataset.udacity_object_dataset import UdacityObjectDataset
    dataset = UdacityObjectDataset(config)

    from object_detection.model.YOLO import YOLO
    model = YOLO(config)

    session = tf.Session()

    iterator = ImageIterator(session, dataset, config, model)

    iterator.create_iterator_from_tfrecords(mode='train')
    iterator.create_iterator(mode='train')


