import os
import yaml

## define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


class ConfigReader(object):

    def __init__(self, config_path='/Users/adam.zvada/Documents/Dev/object-detection/config/test.yml'):

        self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        yaml.add_constructor('!join', join, Loader=yaml.SafeLoader)

        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

            self.info = config['info']
            self.dataset = config['dataset']
            self.hyperparams = config['hyperparams']
            self.model = config['model']
            self.yolo = config['yolo']

    def model_name(self):
        return self.info['model_name']

    # --DATASET--

    def dataset_name(self):
        return self.dataset['name']


    def dataset_path(self):
        path = self.dataset['dataset_path']

        if path == None:
            return None

        return self._absolute_path(path)

    def test_size(self):
        return self.dataset['test_size']

    def num_classes(self):
        return self.dataset['num_classes']

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

        # if not os.path.isabs(path):
        #     print(os.path.join(self.ROOT_DIR, path))
        #     return os.path.join(self.ROOT_DIR, path)

        return path

if __name__ == '__main__':

    config = ConfigReader()

    print(config.model_description())

    print(config.dataset_path())
