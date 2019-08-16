import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from object_detection.utils import image_utils
from sklearn.model_selection import train_test_split
from object_detection.config.config_reader import ConfigReader

class DatasetBase(object):
    """
    Base class for datasets operation.
    """

    def __init__(self, config: ConfigReader):
        self.config = config

        self.df = pd.DataFrame()
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.validate_df = pd.DataFrame()

        # TODO: read anchors from config
        self.anchors_large = [[161.525, 143.0], [58.06666667, 59.36666667], [26.325, 28.4375]]  # preserve aspect ratio
        #self.anchors_large = [[161.19999695, 226.4888916], [57.19999695, 97.06668091], [19.82500076, 63.41111755]]
        self.anchors_large = np.array(self.anchors_large)
        self.anchors_large /= self.config.image_width()

        self.anchors_medium = [[44.60625, 9.66875], [15.925, 12.5125], [8.775, 21.45]]  # preserve aspect ratio
        #self.anchors_medium = [[38.56666565, 32.58666992], [17.0625, 19.5], [8.93748474, 33.27999878]]
        self.anchors_medium = np.array(self.anchors_medium)
        self.anchors_medium /= self.config.image_width()

        self.anchors_small = [[6.0125, 9.99375], [10.075, 5.6875], [4.225, 4.46875]]  # preserve aspect ratio
        #self.anchors_small = [[5.6333313, 17.33332825], [9.5874939, 10.1111145], [4.0625, 7.80000305]]
        self.anchors_small = np.array(self.anchors_small)
        self.anchors_small /= self.config.image_width()

    def load_annotation_df(self):
        raise NotImplemented

    def standardize_classes(self, df):
        raise NotImplemented

    def load_dataset(self):
        print(f'Preparing to load {self.config.dataset_name()} dataset...')

        annotation_df = self.load_annotation_df()

        annotation_df = self.classes_one_hot(annotation_df)

        preprocessed_df = self.yolo_preprocessing(annotation_df)

        train_df, test_df = self.split_dataset(preprocessed_df)

        self.train_df = train_df
        self.test_df = test_df

        print(f'Number of records in test dataset: {len(self.test_df)}')
        print(f'Number of records in train dataset: {len(self.train_df)}')

        # self.generate_tfrecords(train_df, type='train')
        # self.generate_tfrecords(test_df, type='test')

    def classes_one_hot(self, df):

        df_dummies = pd.get_dummies(df['class'])

        if self.config.num_classes() != df_dummies.shape[1]:
            raise Exception('Number of classes does not match.')
            print('Added padding')
            df_dummies.columns = df_dummies.columns.add_categories('other_1')
            df_dummies['other_1'] = 0
            df_dummies.columns = df_dummies.columns.add_categories('other_2')
            df_dummies['other_2'] = 0

        return pd.concat([df, df_dummies], axis=1)

    def split_dataset(self, dataset, test_size=None):

        if test_size is None:
            test_size = self.config.test_size()

        train, test = train_test_split(dataset, test_size=test_size, random_state=42, shuffle=False)

        return train, test

    def set_train_df(self, df):
        self.train_df = df

    def set_test_df(self, df):
        self.test_df = df

    def train_dataset(self):
        return self.train_df

    def test_dataset(self):
        return self.test_df

    def validate_dataset(self):
        return self.validate_df

    def yolo_preprocessing(self, dataset):

        print('Yolo annotation preprocessing...')

        # yolo parameters
        num_classes = self.config.num_classes()
        yolo_image_width = self.config.image_width()

        image_filenames = []
        labels_small = []
        labels_medium = []
        labels_large = []
        for image_filename, boxes in tqdm(dataset.groupby(['image_filename'])):

                label_small = self.yolo_bb_per_image(boxes, self.anchors_small, self.config.num_cells_small(),
                                                     num_classes, yolo_image_width)
                label_medium = self.yolo_bb_per_image(boxes, self.anchors_medium, self.config.num_cells_medium(),
                                                      num_classes, yolo_image_width)
                label_large = self.yolo_bb_per_image(boxes, self.anchors_large, self.config.num_cells_large(),
                                                     num_classes, yolo_image_width)

                image_filenames.append(image_filename)
                labels_small.append(label_small)
                labels_medium.append(label_medium)
                labels_large.append(label_large)

        # preprocessed_df = pd.DataFrame(training_data, columns=['image_filename', 'label'])
        preprocessed_df = pd.DataFrame()
        preprocessed_df['image_filename'] = image_filenames
        preprocessed_df['label_small'] = labels_small
        preprocessed_df['label_medium'] = labels_medium
        preprocessed_df['label_large'] = labels_large

        return preprocessed_df

    # TODO: SIMPLIFY AND CREATE helper functions
    def yolo_bb_per_image(self, boxes, anchors, num_cells, num_classes, yolo_image_size):

        num_anchors = len(anchors)
        cell_size = int(yolo_image_size/num_cells)

        # create zeroed out yolo label
        label = np.zeros(shape=(num_cells, num_cells, num_anchors, 5 + num_classes))

        for traffic_object in boxes.itertuples(index=None, name=None):

            # TRAFFIC OBJECT STRUCTURE
            # traffic_object = ('image_filename', 'image_w', 'image_h', 'image_d',
            #                   'x_min', 'y_min', 'x_max', 'y_max',
            #                   'class', 'dataset_type',  <ONE-HOT-ENCODING>)
            image_shape = traffic_object[1:4]
            bbox = traffic_object[4:8]

            # self.test_plot_regular_img(traffic_object[0], bbox[0], bbox[1], bbox[2], bbox[3], traffic_object[:10])

            # calculate size ratios, img - (w, h, 3)
            width_ratio = yolo_image_size / image_shape[0]
            height_ratio = yolo_image_size / image_shape[1]

            # resize cords
            x_min, y_min, x_max, y_max = image_utils.resize_boxe(bbox, current_size=(image_shape[0], image_shape[1]),
                                                                 target_size=(yolo_image_size, yolo_image_size), keep_ratio=True)

            # self.test_plot_yolo_img(traffic_object[0], x_min, y_min, x_max, y_max, traffic_object[:10])

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
            one_hot = traffic_object[10:]

            for i, (rel_anchor_width, rel_anchor_height) in enumerate(anchors):
                # calculate w,h in respect to anchors size
                a_w = w / (rel_anchor_width * yolo_image_size)  # TODO: is it yolo image size?
                a_h = h / (rel_anchor_height * yolo_image_size)

                label[c_x, c_y, i, :] = np.concatenate((cell_x, cell_y, a_w, a_h, 1, one_hot), axis=None)

                # x_min = cell_size * (cell_x + c_x) - (a_w * rel_anchor_width * yolo_image_size)/2
                # y_min = cell_size * (cell_y + c_y) - (a_h * rel_anchor_height * yolo_image_size)/2
                # x_max = cell_size * (cell_x + c_x) + (a_w * rel_anchor_width * yolo_image_size)/2
                # y_max = cell_size * (cell_y + c_y) + (a_h * rel_anchor_height * yolo_image_size)/2
                #
                # self.test_plot_yolo_img(traffic_object[0], x_min, y_min, x_max, y_max, traffic_object[:10])

        return label

    def test_plot_regular_img(self, img_filename, x_min, x_max, y_min, y_max, one_hot):
        import matplotlib.pyplot as plt
        from object_detection.utils import image_utils
        import cv2

        image = cv2.imread(img_filename)

        image = image_utils.draw_boxes_PIL(image , boxes=[(x_min, x_max, y_min, y_max)], scores=[1],
                                           classes=one_hot)
        plt.imshow(image)
        plt.show()

    def test_plot_yolo_img(self, img_filename, x_min, x_max, y_min, y_max, one_hot):
        import matplotlib.pyplot as plt
        from object_detection.utils import image_utils
        import cv2

        image = cv2.imread(img_filename)

        #resized_img = cv2.resize(image, (416, 416), interpolation=cv2.INTER_NEAREST)
        resized_img = image_utils.letterbox_image_2(image, (416, 416))
        image = image_utils.draw_boxes_PIL(resized_img, boxes=[(x_min, x_max, y_min, y_max)], scores=[1],
                                           classes=one_hot)
        plt.imshow(image)
        plt.show()

    def dataset_tfrecords(self):

        self.generate_tfrecords(self.train_df, type='train')
        self.generate_tfrecords(self.test_df, type='test')

    def generate_tfrecords(self, df, type='train'):

        image_filenames = np.array(df['image_filename'].values.tolist())
        labels = np.array(df['label'].values.tolist(), dtype=np.float32)

        #print({self.config.dataset_path())

        save_path = f'{self.config.dataset_path()}/{self.config.dataset_name()}_{type}.tfrecords'
        with tf.python_io.TFRecordWriter(save_path) as writer:
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


