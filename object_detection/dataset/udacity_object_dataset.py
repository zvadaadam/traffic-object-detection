import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from object_detection.dataset.dataset_base import DatasetBase
from object_detection.utils import image_utils
from object_detection.utils import yolo_utils
from object_detection.config.config_reader import ConfigReader


class UdacityObjectDataset(DatasetBase):

    def __init__(self, config: ConfigReader):
        super(UdacityObjectDataset, self).__init__(config)

        # TODO: read from config
        self.anchors = [[0.05524553571428571, 0.045619419642857144],
                        [0.022042410714285716, 0.029296875],
                        [0.13853236607142858, 0.10407366071428571]]

        self.dataset_path = self.config.udacity_dataset_path()
        self.annotation_path = os.path.join(self.dataset_path, 'labels.csv')

        #self.load_dataset()

    def load_annotation_df(self):

        print(f'Loading annotation dataset {self.config.udacity_dataset_name()}')

        with open(self.annotation_path, 'r') as f:
            data = f.readlines()
        data = [item.split() for item in data]

        records = []

        for data_row in data:
            image_filename = data_row[0]
            # TODO: investigate if all has this shape or load directly from the image
            image_shape_w = 1920
            image_shape_h = 1200
            image_shape_d = 3

            x_min = data_row[1]
            y_min = data_row[2]
            x_max = data_row[3]
            y_max = data_row[4]

            object_class = data_row[6]
            object_class = object_class.replace('\"', '')

            record = [image_filename, image_shape_w, image_shape_h, image_shape_d,
                      x_min, y_min, x_max, y_max,
                      object_class, 'udacity']
            records.append(record)

        df = pd.DataFrame(records, columns=['image_filename', 'image_w', 'image_h', 'image_d',
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

    # # TODO: delete after check if working
    # def _yolo_preprocessing(self, dataset):
    #
    #     # yolo parameters
    #     num_classes = self.config.num_classes()
    #     yolo_image_width = self.config.image_width()
    #     yolo_image_height = self.config.image_height()
    #
    #     image_size = int(yolo_image_width)
    #     num_grids = self.config.grid_size()
    #     cell_size = int(image_size/num_grids)
    #     num_anchors = len(self.anchors)
    #
    #     df = pd.DataFrame()
    #
    #     labels = []
    #     images = []
    #
    #     # TODO: SIMPLIFY AND CREATE FUNCS
    #     for frame, boxes in tqdm(dataset.groupby(['frame'])):
    #         # load images
    #         img = image_utils.load_img(self.config.dataset_path(), frame)
    #         original_image_shape = img.shape
    #
    #         # resize image to yolo wxh
    #         resized_img = image_utils.resize_image(img, self.config.image_height(), self.config.image_height())
    #         resized_image_shape = resized_img.shape
    #
    #         images.append(resized_img)
    #
    #         # create zeroed out yolo label
    #         label = np.zeros(shape=(num_grids, num_grids, num_anchors, 5 + num_classes))
    #
    #         for box in boxes.itertuples(index=None, name=None):
    #
    #             # calculate size ratios, img - (h, w, 3)
    #             width_ratio = resized_image_shape[1]/original_image_shape[1]
    #             height_ratio = resized_image_shape[0]/original_image_shape[0]
    #
    #             # resize cords
    #             x_min, x_max = width_ratio * float(box[1]), width_ratio * float(box[3])
    #             y_min, y_max = height_ratio * float(box[2]), height_ratio * float(box[4])
    #
    #             #print(f'{x_min}, {y_min}, {x_max}, {y_max}')
    #
    #             #image_utils.plot_img(image_utils.add_bb_to_img(resized_img, int(x_min), int(y_min), int(x_max), int(y_max)))
    #
    #             if (x_max < x_min or y_max < y_min):
    #                 raise Exception('Invalid groud truth data, Max < Min')
    #
    #             # convert to x, y, w, h
    #             x = (x_min + x_max)/2
    #             y = (y_min + y_max)/2
    #             w = x_max - x_min
    #             h = y_max - y_min
    #
    #             # make x, y relative to its cell
    #             origin_box_x = int(x / cell_size) * cell_size
    #             origin_box_y = int(y / cell_size) * cell_size
    #
    #             cell_x = (x - origin_box_x) / cell_size
    #             cell_y = (y - origin_box_y) / cell_size
    #
    #             # cell index
    #             g_x, g_y = int(origin_box_x / cell_size), int(origin_box_y / cell_size)
    #
    #             # class data
    #             one_hot = box[7:]
    #
    #             # x_min = int( ((cell_x * 64) + g_x * 64) - w/2)
    #             # y_min = int( ((cell_y * 64) + g_y * 64) - h/2)
    #             #
    #             # x_max = int( ((cell_x * 64) + g_x * 64) + w/2)
    #             # y_max = int( ((cell_y * 64) + g_y * 64) + h/2)
    #
    #             #image_utils.plot_img(image_utils.add_bb_to_img(resized_img, x_min, y_min, x_max, y_max))
    #
    #             for i, (rel_anchor_width, rel_anchor_height) in enumerate(self.anchors):
    #
    #                 # calculate w,h in respect to anchors size
    #                 a_w = w / (rel_anchor_width * image_size)
    #                 a_h = h / (rel_anchor_height * image_size)
    #
    #                 label[g_x, g_y, i, :] = np.concatenate((cell_x, cell_y, a_w, a_h, 1, one_hot), axis=None)
    #
    #                 #print(f'{cell_x}, {cell_y}, {a_w}, {a_h}')
    #
    #                 x_min = int((((cell_x*cell_size) + g_x*cell_size) - (a_w * (rel_anchor_width * image_size)/2)))
    #                 y_min = int((((cell_y*cell_size) + g_y*cell_size) - (a_h * (rel_anchor_height * image_size)/2)))
    #
    #                 x_max = int((((cell_x*cell_size) + g_x*cell_size) + (a_w * (rel_anchor_width * image_size)/2)))
    #                 y_max = int((((cell_y*cell_size) + g_y*cell_size) + (a_h * (rel_anchor_height * image_size)/2)))
    #
    #                 #print(f'{x_min}, {y_min}, {x_max}, {y_max}')
    #
    #                 #image_utils.plot_img(image_utils.add_bb_to_img(resized_img, x_min, y_min, x_max, y_max))
    #
    #
    #         labels.append(label)
    #
    #         # x_min = int((x - w/2)*448)
    #         # y_min = int((y - h/2)*448)
    #         #
    #         # x_max = int((x + w/2)*448)
    #         # y_max = int((y + h/2)*448)
    #         #
    #         # image_utils.plot_img(image_utils.add_bb_to_img(img, x_min, y_min, x_max, y_max))
    #
    #     df['image'] = images
    #     df['label'] = labels
    #
    #     self.save_to_pickle(df)
    #
    #     return df
    #
    # def generate_tfrecords(self, df, type='train'):
    #
    #     images = np.array(df['image'].values.tolist(), dtype=np.float32)
    #     labels = np.array(df['label'].values.tolist(), dtype=np.float32)
    #
    #     with tf.python_io.TFRecordWriter(f'{self.config.root_dataset()}/udacity_{type}.tfrecords') as writer:
    #
    #         for image, labels in tqdm(zip(images, labels)):
    #             encoded_image = cv2.imencode('.jpg', image)[1].tostring()
    #
    #             # flat_labels = labels.flatten()
    #
    #             example = tf.train.Example(features=tf.train.Features(
    #                 feature={
    #                     'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(encoded_image)])),
    #                     'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.tostring()]))
    #                     # 'labels': tf.train.Feature(float_list=tf.train.FloatList(value=flat_labels))
    #                 }))
    #
    #             writer.write(example.SerializeToString())
    #
    # def save_to_pickle(self, df):
    #     df.to_pickle('udacity_dataset_500.pkl')
    #
    # def load_pickle(self, path):
    #     return pd.read_pickle(path)

if __name__ == '__main__':

    from object_detection.config.config_reader import ConfigReader

    config = ConfigReader()

    dataset = UdacityObjectDataset(config)

    # frame = np.asarray(dataset.labels_df['frame'])[0]
    #
    # x_min = np.asarray(dataset.labels_df['x_min'])[0]
    # y_min = np.asarray(dataset.labels_df['y_min'])[0]
    #
    # x_max = np.asarray(dataset.labels_df['x_max'])[0]
    # y_max = np.asarray(dataset.labels_df['y_max'])[0]
    #
    # img = image_utils.load_img(config.dataset_path(), frame)
    # image_utils.plot_img(img)
    #
    # x, y, w, h = yolo_utils.from_cord_to_yolo(x_min, y_min, x_max, y_max, img.shape)
    # #image_utils.plot_img(image_utils.add_bb_to_img(img, x_min, y_min, x_max, y_max))
    #
    # img, [(x, y, w, h)] = yolo_utils.img_to_yolo_shape(img, [(x, y, w, h)])
    #
    # print(img.shape)
    #
    # x_min, y_min, x_max, y_max = yolo_utils.from_yolo_to_cord((x, y, w, h), img.shape)
    #
    # image_utils.plot_img(image_utils.add_bb_to_img(img, x_min, y_min, x_max, y_max))


    # grid_size = config.grid_size()
    # B = config.boxes_per_cell()
    # num_classes = config.num_classes()
    #
    # df = pd.DataFrame()
    #
    # labels = []
    # images = []
    #
    # for frame, boxes in tqdm(dataset.labels_df.groupby(['frame'])):
    #
    #     # load images
    #     img = image_utils.load_img(config.dataset_path(), frame)
    #     original_image_shape = img.shape
    #
    #     # create zeroed out yolo label
    #     label = np.zeros(shape=(grid_size, grid_size, B * 5 + num_classes))
    #
    #     for box in boxes.itertuples(index=None, name=None):
    #         # convert box cordinates to yolo format cordinates
    #         x, y, w, h = yolo_utils.from_cord_to_yolo(box[1], box[2], box[3], box[4], img.shape)
    #
    #         # resize the cordinates to yolo size
    #         [(x, y, w, h)] = yolo_utils.yolo_cords([(x, y, w, h)], original_image_shape)
    #
    #         # one hot encoding for box classification
    #         one_hot = box[7:]
    #
    #         g_x, g_y = yolo_utils.grid_index(x, y)
    #
    #         label[g_x, g_y, :] = np.concatenate((x, y, w, h, 1, one_hot), axis=None)
    #
    #     labels.append(label)
    #
    #     img = yolo_utils.img_to_yolo_shape(img)
    #     images.append(img)
    #
    # print(len(labels))
    # print(len(images))
    #
    # df['image'] = images
    # df['label'] = labels
    #
    # # df.to_csv('yolo_dataset.csv')
    # # df.to_pickle('yolo_dataset.pkl')



















