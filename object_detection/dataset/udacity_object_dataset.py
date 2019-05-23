import pandas as pd
import numpy as np
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
        self.anchors = [[0.05524553571428571, 0.045619419642857144], [0.022042410714285716, 0.029296875], [0.13853236607142858, 0.10407366071428571]]

        self.load_dataset()

        print(self.df.isnull().values.any())
        #self.load_dataset_from_pickle('udacity_dataset_500.pkl')

    def load_dataset_from_pickle(self, path):
        df = self.load_pickle(path)

        train_df, test_df = train_test_split(df, test_size=self.config.test_size())

        self.set_train_df(train_df)
        self.set_test_df(test_df)

    def load_dataset(self):
        df = self.load_udacity_dataset()

        train_df, test_df = train_test_split(df, test_size=self.config.test_size())

        self.set_train_df(train_df)
        self.set_test_df(test_df)

    def load_udacity_dataset(self):

        print(f'Loading {self.config.dataset_name()} dataset...')
        dataset_path = self.config.dataset_path() + '/labels.csv'

        with open(dataset_path, 'r') as f:
            data = f.readlines()
        data = [item.split() for item in data]

        return self.create_labels_dataframe(data)

    def create_labels_dataframe(self, data):

        df = pd.DataFrame()

        frames = []
        x_min = []
        y_min = []
        x_max = []
        y_max = []
        occlusions = []
        labels = []

        #for data_row in data[0:500]:
        for data_row in data:
            frames.append(data_row[0])
            x_min.append(data_row[1])
            y_min.append(data_row[2])
            x_max.append(data_row[3])
            y_max.append(data_row[4])
            occlusions.append(data_row[5])

            label = data_row[6]
            label = label.replace('\"', '')
            labels.append(label)

        df['frame'] = frames
        df['frame'] = df['frame'].astype(str)

        df['x_min'] = x_min
        df['x_min'] = df['x_min'].astype(np.float16)

        df['y_min'] = y_min
        df['y_min'] = df['y_min'].astype(np.float16)

        df['x_max'] = x_max
        df['x_max'] = df['x_max'].astype(np.float16)

        df['y_max'] = y_max
        df['y_max'] = df['y_max'].astype(np.float16)

        df['occlusion'] = occlusions
        df['occlusion'] = df['occlusion'].astype(bool)

        df['label'] = labels
        df['label'] = df['label'].astype(str)
        df['label'] = df['label'].astype('category')

        df_dummies = pd.get_dummies(df['label'], prefix='category')
        df = pd.concat([df, df_dummies], axis=1)

        self.df_true = df

        return self.yolo_dataset_format(df)

    def yolo_dataset_format(self, dataset):

        # YOLO: Input image: 448x448, output: 7x7x(B*5 + num_classes)

        # yolo parameters
        num_classes = self.config.num_classes()
        yolo_image_width = self.config.image_width()
        yolo_image_height = self.config.image_height()

        if yolo_image_width != yolo_image_height:
            raise Exception('YOLO supports images that are rectangles, if you don\'t like, try to change it...')

        image_size = int(yolo_image_width)
        num_grids = self.config.grid_size()
        cell_size = int(image_size/num_grids)
        num_anchors = len(self.anchors)

        df = pd.DataFrame()

        labels = []
        images = []

        # TODO: SIMPLIFY AND CREATE FUNCS
        for frame, boxes in tqdm(dataset.groupby(['frame'])):
            # load images
            img = image_utils.load_img(self.config.dataset_path(), frame)
            original_image_shape = img.shape

            # resize image to yolo wxh
            resized_img = image_utils.resize_image(img, self.config.image_height(), self.config.image_height())
            resized_image_shape = resized_img.shape

            images.append(resized_img)

            # create zeroed out yolo label
            label = np.zeros(shape=(num_grids, num_grids, num_anchors, 5 + num_classes))

            for box in boxes.itertuples(index=None, name=None):

                # calculate size ratios, img - (h, w, 3)
                width_ratio = resized_image_shape[1]/original_image_shape[1]
                height_ratio = resized_image_shape[0]/original_image_shape[0]

                # resize cords
                x_min, x_max = width_ratio * float(box[1]), width_ratio * float(box[3])
                y_min, y_max = height_ratio * float(box[2]), height_ratio * float(box[4])

                #print(f'{x_min}, {y_min}, {x_max}, {y_max}')

                #image_utils.plot_img(image_utils.add_bb_to_img(resized_img, int(x_min), int(y_min), int(x_max), int(y_max)))

                if (x_max < x_min or y_max < y_min):
                    raise Exception('Invalid groud truth data, Max < Min')

                # convert to x, y, w, h
                x = (x_min + x_max)/2
                y = (y_min + y_max)/2
                w = x_max - x_min
                h = y_max - y_min

                # make x, y relative to its cell
                origin_box_x = int(x / cell_size) * cell_size
                origin_box_y = int(y / cell_size) * cell_size

                cell_x = (x - origin_box_x) / cell_size
                cell_y = (y - origin_box_y) / cell_size

                # cell index
                g_x, g_y = int(origin_box_x / cell_size), int(origin_box_y / cell_size)

                # class data
                one_hot = box[7:]

                # x_min = int( ((cell_x * 64) + g_x * 64) - w/2)
                # y_min = int( ((cell_y * 64) + g_y * 64) - h/2)
                #
                # x_max = int( ((cell_x * 64) + g_x * 64) + w/2)
                # y_max = int( ((cell_y * 64) + g_y * 64) + h/2)

                #image_utils.plot_img(image_utils.add_bb_to_img(resized_img, x_min, y_min, x_max, y_max))

                for i, (rel_anchor_width, rel_anchor_height) in enumerate(self.anchors):

                    # calculate w,h in respect to anchors size
                    a_w = w / (rel_anchor_width * image_size)
                    a_h = h / (rel_anchor_height * image_size)

                    label[g_x, g_y, i, :] = np.concatenate((cell_x, cell_y, a_w, a_h, 1, one_hot), axis=None)

                    #print(f'{cell_x}, {cell_y}, {a_w}, {a_h}')

                    x_min = int((((cell_x*64) + g_x*64) - (a_w * (rel_anchor_width * image_size)/2)))
                    y_min = int((((cell_y*64) + g_y*64) - (a_h * (rel_anchor_height * image_size)/2)))

                    x_max = int((((cell_x*64) + g_x*64) + (a_w * (rel_anchor_width * image_size)/2)))
                    y_max = int((((cell_y*64) + g_y*64) + (a_h * (rel_anchor_height * image_size)/2)))

                    #print(f'{x_min}, {y_min}, {x_max}, {y_max}')

                    #image_utils.plot_img(image_utils.add_bb_to_img(resized_img, x_min, y_min, x_max, y_max))


            labels.append(label)

            # x_min = int((x - w/2)*448)
            # y_min = int((y - h/2)*448)
            #
            # x_max = int((x + w/2)*448)
            # y_max = int((y + h/2)*448)
            #
            # image_utils.plot_img(image_utils.add_bb_to_img(img, x_min, y_min, x_max, y_max))

        df['image'] = images
        df['label'] = labels

        self.save_to_pickle(df)

        return df

    def save_to_pickle(self, df):
        df.to_pickle('udacity_dataset_500.pkl')

    def load_pickle(self, path):
        return pd.read_pickle(path)

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



















