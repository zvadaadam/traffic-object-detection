import os
import pandas as pd
import numpy as np
import json
import tensorflow as tf
import cv2
import xmltodict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from object_detection.dataset.dataset_base import DatasetBase
from object_detection.utils import image_utils
from object_detection.utils import yolo_utils
from object_detection.config.config_reader import ConfigReader


class BddDataset(DatasetBase):
    """
    Berkeley Deep Drive Dataset
    """

    def __init__(self, config: ConfigReader):
        super(BddDataset, self).__init__(config)

        # TODO: read from config
        # TODO: calculate ancgores for bdd
        # self.anchors = [[0.05524553571428571, 0.045619419642857144],
        #                 [0.022042410714285716, 0.029296875],
        #                 [0.13853236607142858, 0.10407366071428571]]

        self.dataset_path = self.config.bdd_dataset_path()

        self.annotations_path = os.path.join(self.dataset_path, 'labels')
        self.image_path = os.path.join(self.dataset_path, 'images/100k')

        # self.load_dataset(create_tfrecord, recalculate_labels)

        # self.load_dataset_from_pickle('udacity_dataset_500.pkl')
        # self.generate_tfrecords(self.train_df, type='train')
        # self.generate_tfrecords(self.test_df, type='test')

    def load_annotation_df(self):

        print(f'Loading annotation dataset {self.config.bdd_dataset_name()}')

        objects_records = []
        annotations_jsons = os.listdir(self.annotations_path)

        for annotations_json in annotations_jsons:
            filename = os.path.join(self.annotations_path, annotations_json)
            objects_records = objects_records + self.parse_annotation_filename(filename)

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

        df = self.standardize_classes(df)

        print(df['class'].unique())

        return df

    def standardize_classes(self, df):

        df['class'] = df['class'].astype(str)

        if (df['class'] == 'traffic sign').any():
            df.loc[df['class'] == 'traffic sign', 'class'] = 'trafficSign'

        if (df['class'] == 'traffic light').any():
            df.loc[df['class'] == 'traffic light', 'class'] = 'trafficLight'

        if (df['class'] == 'rider').any():
            df.loc[df['class'] == 'rider', 'class'] = 'person'

        if (df['class'] == 'motor').any():
            df.loc[df['class'] == 'motor', 'class'] = 'motorbike'

        df['class'] = df['class'].astype('category')

        return df

    def parse_annotation_filename(self, filename):

        records = []

        with open(filename) as json_file:
            annotations = json.load(json_file)

        if 'train' in filename:
            print('Processing Training')
            image_path = os.path.join(self.image_path, 'train')
        else:
            print('Processing Validation')
            image_path = os.path.join(self.image_path, 'val')

        #for image_annotations in tqdm(annotations[:500]):
        for image_annotations in tqdm(annotations):
            image_filename = os.path.join(image_path, image_annotations['name'])

            for label in image_annotations['labels']:
                object_class = label['category']
                if object_class == 'drivable area' or object_class == 'lane':
                    continue

                x_min = label['box2d']['x1']
                y_min = label['box2d']['y1']
                x_max = label['box2d']['x2']
                y_max = label['box2d']['y2']

                image_shape_w = 1280
                image_shape_h = 720
                image_shape_d = 3

                record = [image_filename, image_shape_w, image_shape_h, image_shape_d, x_min, y_min, x_max, y_max,
                          object_class, 'bdd']
                records.append(record)

        return records

if __name__ == '__main__':

    from object_detection.config.config_reader import ConfigReader

    config = ConfigReader()

    dataset = BddDataset(config)

    dataset.load_dataset()

    test_df = dataset.test_dataset()
    image_filenames = test_df['image_filename'].values.tolist()[90:190]

