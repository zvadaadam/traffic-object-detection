import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import xmltodict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from object_detection.dataset.dataset_base import DatasetBase
from object_detection.utils import image_utils
from object_detection.utils import yolo_utils
from object_detection.config.config_reader import ConfigReader


class RovitDataset(DatasetBase):

    def __init__(self, config: ConfigReader):
        super(RovitDataset, self).__init__(config)

        # TODO: read from config
        self.anchors = [[0.05524553571428571, 0.045619419642857144],
                        [0.022042410714285716, 0.029296875],
                        [0.13853236607142858, 0.10407366071428571]]

        self.dataset_path = self.config.dataset_path()
        self.annotations_path = os.path.join(self.dataset_path, 'Annotations')
        self.image_path = os.path.join(self.dataset_path, 'JPEGImages')

        # self.load_dataset(create_tfrecord, recalculate_labels)

        # self.load_dataset_from_pickle('udacity_dataset_500.pkl')
        # self.generate_tfrecords(self.train_df, type='train')
        # self.generate_tfrecords(self.test_df, type='test')

    def load_dataset(self):

        print(f'Loading RoVit dataset...')
        df = self.create_training_df()

        train_df, test_df = train_test_split(df, test_size=self.config.test_size(), random_state=42)

        self.set_train_df(train_df)
        self.set_test_df(test_df)

    def create_training_df(self):

        objects_records = []
        annotations_filenames = os.listdir(self.annotations_path)
        for annotations_filename in tqdm(annotations_filenames):
            objects_records = objects_records + self.parse_annotation_file(annotations_filename)

        df = pd.DataFrame(objects_records, columns=['image_filename', 'image_w', 'image_h', 'image_d',
                                                    'x_min', 'y_min', 'x_max', 'y_max',
                                                    'class'])

        df['image_filename'] = df['image_filename'].astype(str)
        df['image_w'] = df['image_w'].astype(np.float16)
        df['image_h'] = df['image_h'].astype(np.float16)
        df['image_d'] = df['image_d'].astype(np.float16)
        df['x_min'] = df['x_min'].astype(np.float16)
        df['y_min'] = df['y_min'].astype(np.float16)
        df['x_max'] = df['x_max'].astype(np.float16)
        df['y_max'] = df['y_max'].astype(np.float16)
        df['class'] = df['class'].astype(str)
        df['class'] = df['class'].astype('category')

        # one hot encoding for classes
        df_dummies = pd.get_dummies(df['class'])
        df = pd.concat([df, df_dummies], axis=1)

        df.to_csv('rovic-dataset.csv')

        return self.yolo_preprocessing(df)

    def parse_annotation_file(self, filename):

        records = []

        annotation_file = os.path.join(self.annotations_path, filename)

        with open(annotation_file) as fd:
            doc = xmltodict.parse(fd.read(), force_list={'object'})['annotation']

        image_filename = doc['filename']
        image_shape_w = doc['size']['width']
        image_shape_h = doc['size']['height']
        image_shape_d = doc['size']['depth']

        try:
            traffic_objects = doc['object']
        except:
            return []

        for traffic_object in traffic_objects:
            object_class = traffic_object['name']
            x_min = traffic_object['bndbox']['xmin']
            y_min = traffic_object['bndbox']['ymin']
            x_max = traffic_object['bndbox']['xmax']
            y_max = traffic_object['bndbox']['ymax']

            record = [image_filename, image_shape_w, image_shape_h, image_shape_d,
                      x_min, y_min, x_max, y_max,
                      object_class]
            records.append(record)

        return records

    def yolo_preprocessing(self, dataset):

        # yolo parameters
        num_classes = self.config.num_classes()
        yolo_image_width = self.config.image_width()
        yolo_image_height = self.config.image_height()
        num_cells = self.config.grid_size()
        num_anchors = self.config.num_anchors()

        image_size = int(yolo_image_width)
        cell_size = int(image_size / num_cells)

        training_data = []
        # TODO: SIMPLIFY AND CREATE FUNCS
        for image_filename, boxes in tqdm(dataset.groupby(['image_filename'])):

            # create zeroed out yolo label
            label = np.zeros(shape=(num_cells, num_cells, num_anchors, 5 + num_classes))

            for traffic_object in boxes.itertuples(index=None, name=None):

                # TRAFFIC OBJECT STRUTURE
                # traffic_object = ('image_filename', 'image_w', 'image_h', 'image_d',
                #                   'x_min', 'y_min', 'x_max', 'y_max',
                #                   'class', <ONE-HOT-ENCODING>)
                image_shape = traffic_object[1:4]
                bbox = traffic_object[4:8]

                # calculate size ratios, img - (w, h, 3)
                width_ratio = yolo_image_width / image_shape[0]
                height_ratio = yolo_image_width / image_shape[1]

                # resize cords
                x_min, x_max = width_ratio * float(bbox[0]), width_ratio * float(bbox[2])
                y_min, y_max = height_ratio * float(bbox[1]), height_ratio * float(bbox[3])

                # convert to x, y, w, h
                x = (x_min + x_max) / 2
                y = (y_min + y_max) / 2
                w = x_max - x_min
                h = y_max - y_min

                # make x, y relative to its cell origin
                origin_box_x = int(x / cell_size) * cell_size
                origin_box_y = int(y / cell_size) * cell_size
                cell_x = (x - origin_box_x) / cell_size
                cell_y = (y - origin_box_y) / cell_size

                # cell index
                c_x, c_y = int(origin_box_x / cell_size), int(origin_box_y / cell_size)

                # class data
                one_hot = traffic_object[9:]

                for i, (rel_anchor_width, rel_anchor_height) in enumerate(self.anchors):
                    # calculate w,h in respect to anchors size
                    a_w = w / (rel_anchor_width * image_size)
                    a_h = h / (rel_anchor_height * image_size)

                    label[c_x, c_y, i, :] = np.concatenate((cell_x, cell_y, a_w, a_h, 1, one_hot), axis=None)

                training_data.append([image_filename, label])

        preprocessed_df = pd.DataFrame(training_data, columns=['image_filename', 'label'])

        return preprocessed_df

    def dataset_tfrecords(self):

        self.generate_tfrecords(self.train_df, type='train')
        self.generate_tfrecords(self.test_df, type='test')

    def generate_tfrecords(self, df, type='train'):

        image_filenames = np.array(df['image_filename'].values.tolist())
        labels = np.array(df['label'].values.tolist(), dtype=np.float32)

        with tf.python_io.TFRecordWriter(f'{self.dataset_path}/rovit_{type}.tfrecords') as writer:
            for image_filename, labels in tqdm(zip(image_filenames, labels)):
                #             img_path = os.path.join(image_path, image_filename)
                #             img_raw = tf.read_file(img_path)
                #             img = tf.image.decode_image(img_raw)
                #             img = tf.image.resize_images(img_tensor, [YOLO_WIDTH, YOLO_HEIGHT])

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image_filename': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[image_filename.tostring()])),
                        'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.tostring()]))
                    }))

                writer.write(example.SerializeToString())
