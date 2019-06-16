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

        self.dataset_path = self.config.rovit_dataset_path()
        self.annotations_path = os.path.join(self.dataset_path, 'Annotations')
        self.image_path = os.path.join(self.dataset_path, 'JPEGImages')

        # self.load_dataset(create_tfrecord, recalculate_labels)

        # self.load_dataset_from_pickle('udacity_dataset_500.pkl')
        # self.generate_tfrecords(self.train_df, type='train')
        # self.generate_tfrecords(self.test_df, type='test')

    def load_annotation_df(self):

        objects_records = []
        annotations_filenames = os.listdir(self.annotations_path)
        for annotations_filename in tqdm(annotations_filenames):
            objects_records = objects_records + self.parse_annotation_file(annotations_filename)

        df = pd.DataFrame(objects_records, columns=['image_filename', 'image_w', 'image_h', 'image_d',
                                                    'x_min', 'y_min', 'x_max', 'y_max',
                                                    'class', 'dataset_name'])

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
        df['dataset_name'] = df['dataset_name'].astype(str)


        return df

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
                      object_class, 'rovit']
            records.append(record)

        return records
