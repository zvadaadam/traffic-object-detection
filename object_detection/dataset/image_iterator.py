import tensorflow as tf
import numpy as np
from object_detection.config.config_reader import ConfigReader
from object_detection.dataset.dataset_base import DatasetBase
from object_detection.model.base_model import ModelBase


class ImageIterator(object):


    def __init__(self, session: tf.Session, model, dataset: DatasetBase, config: ConfigReader):
        self.session = session
        self.model = model
        self.dataset = dataset
        self.config = config

        self.handle_placeholder = tf.placeholder(tf.string, shape=[])


    def create_iterator(self, mode='train'):

        dataset = tf.data.Dataset.from_tensor_slices((self.model.x, self.model.y))

        dataset = dataset.batch(self.config.batch_size(), drop_remainder=True).repeat()

        dataset_iterator = dataset.make_initializable_iterator()

        generic_iterator = tf.data.Iterator.from_string_handle(self.handle_placeholder, dataset.output_types,
                                                               dataset.output_shapes, dataset.output_classes)

        dataset_handle = self.session.run(dataset_iterator.string_handle())

        x, y = generic_iterator.get_next()
        inputs = {
            'x': x,
            'y': y
        }

        if mode == 'train':
            df = self.dataset.train_dataset()
        else:
            df = self.dataset.test_dataset()

        images = np.array(df['image'].values.tolist(), dtype=np.float32)
        labels = np.array(df['label'].values.tolist())

        # TODO: MOVE NORMALIZATION
        for i in range(0, len(images)):
            img = images[i]
            img = img.astype('float32')
            if img.max() > 1.0:
                img /= 255.0

            images[i] = img

        feed = {
            self.model.x: images,
            self.model.y: labels
        }
        self.session.run(dataset_iterator.initializer, feed_dict=feed)

        return inputs, dataset_handle

    def normalization(self, img):

        img_norm = img - img.mean()

        img_norm = img_norm / img_norm.max()

        return img_norm
