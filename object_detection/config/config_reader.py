import os
import yaml

## define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


class ConfigReader(object):

    def __init__(self, config_path='/Users/adam.zvada/Documents/Dev/object-detection/config/yolo.yml'):

        self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        yaml.add_constructor('!join', join, Loader=yaml.SafeLoader)

        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

            self.info = config['info']
            self.dataset = config['dataset']
            self.udacity = config['udacity']
            self.rovit = config['rovit']
            self.bdd = config['bdd']
            self.hyperparams = config['hyperparams']
            self.model = config['model']
            self.yolo = config['yolo']

    def model_name(self):
        return self.info['model_name']

    # --DATASET--

    def dataset_name(self):
        return self.dataset['name']

    def root_dataset(self):
        path = self.dataset['root_dataset']

        if path == None:
            return None

        return self._absolute_path(path)


    def dataset_path(self):
        path = self.dataset['dataset_path']

        if path == None:
            return None

        return self._absolute_path(path)

    def tfrecords_train_path(self):
        path = self.dataset['tfrecords_train_path']

        if path == None:
            return None

        return self._absolute_path(path)

    def tfrecords_test_path(self):
        path = self.dataset['tfrecords_test_path']

        if path == None:
            return None

        return self._absolute_path(path)

    def test_size(self):
        return self.dataset['test_size']

    def num_classes(self):
        return self.dataset['num_classes']

    def preserve_aspect_ratio(self):
        return self.dataset['preserve_aspect_ratio']

    # --DATASET--

    def udacity_dataset_name(self):
        return self.udacity['name']

    def udacity_root_dataset(self):
        path = self.udacity['root_dataset']

        if path == None:
            return None

        return self._absolute_path(path)


    def udacity_dataset_path(self):
        path = self.udacity['dataset_path']

        if path == None:
            return None

        return self._absolute_path(path)

    def udacity_tfrecords_train_path(self):
        path = self.udacity['tfrecords_train_path']

        if path == None:
            return None

        return self._absolute_path(path)

    def udacity_tfrecords_test_path(self):
        path = self.udacity['tfrecords_test_path']

        if path == None:
            return None

        return self._absolute_path(path)

    def udacity_test_size(self):
        return self.udacity['test_size']

    def udacity_num_classes(self):
        return self.udacity['num_classes']

        # --BERKLEY DEEP DRIVE--

    def bdd_dataset_name(self):
        return self.bdd['name']

    def bdd_root_dataset(self):
        path = self.bdd['root_dataset']

        if path == None:
            return None

        return self._absolute_path(path)

    def bdd_dataset_path(self):
        path = self.bdd['dataset_path']

        if path == None:
            return None

        return self._absolute_path(path)

    def bdd_tfrecords_train_path(self):
        path = self.bdd['tfrecords_train_path']

        if path == None:
            return None

        return self._absolute_path(path)

    def bdd_tfrecords_test_path(self):
        path = self.rovit['tfrecords_test_path']

        if path == None:
            return None

        return self._absolute_path(path)

    def bdd_test_size(self):
        return self.bdd['test_size']

    def bdd_num_classes(self):
        return self.bdd['num_classes']

    # --ROVIT--

    def rovit_dataset_name(self):
        return self.rovit['name']

    def rovit_root_dataset(self):
        path = self.rovit['root_dataset']

        if path == None:
            return None

        return self._absolute_path(path)


    def rovit_dataset_path(self):
        path = self.rovit['dataset_path']

        if path == None:
            return None

        return self._absolute_path(path)

    def rovit_tfrecords_train_path(self):
        path = self.rovit['tfrecords_train_path']

        if path == None:
            return None

        return self._absolute_path(path)

    def rovit_tfrecords_test_path(self):
        path = self.rovit['tfrecords_test_path']

        if path == None:
            return None

        return self._absolute_path(path)

    def rovit_test_size(self):
        return self.rovit['test_size']

    def rovit_num_classes(self):
        return self.rovit['num_classes']

    # --YOLO--

    def image_height(self):
        return self.yolo['image_height']

    def image_width(self):
        return self.yolo['image_width']

    def grid_size(self):
        return self.yolo['grid_size']

    def boxes_per_cell(self):
        return self.yolo['boxes_per_cell']

    def num_anchors(self):
        return self.yolo['num_anchors']

    def anchores(self):
        return self.yolo['anchores']

    def num_cells_small(self):
        return self.yolo['num_cells_small']

    def num_cells_medium(self):
        return self.yolo['num_cells_medium']

    def num_cells_large(self):
        return self.yolo['num_cells_large']

    def num_scales(self):
        return self.yolo['num_scales']

    # --HYPERPARAMETERS--

    def feature_size(self):
        return self.hyperparams['feature_size']

    def batch_size(self):
        return self.hyperparams['batch_size']

    def num_layers(self):
        return self.hyperparams['num_layers']

    def num_classes(self):
        return self.hyperparams['num_classes']

    def num_epoches(self):
        return self.hyperparams['num_epoches']

    def num_iterations(self):
        return self.hyperparams['num_iterations']

    def learning_rate(self):
        return self.hyperparams['learning_rate']

    # --MODEL--

    def tensorboard_path(self):
        path = self.model['tensorboard_path']

        return self._absolute_path(path)

    def trained_model_path(self):
        path = self.model['trained_path']

        return self._absolute_path(path)

    def model_description(self):
        return self.model['model_description']

    def restore_trained_model(self):
        path = self.model['restore_trained_model']

        if path == None:
            return None

        return self._absolute_path(path)

    def _absolute_path(self, path):

        if not os.path.isabs(path):
            return os.path.join(self.ROOT_DIR, path)

        return path

if __name__ == '__main__':

    config = ConfigReader()

    print(config.model_description())

    print(config.dataset_path())
