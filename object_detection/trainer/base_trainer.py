import tensorflow as tf
from tqdm import trange

class BaseTrain(object):
    """
    Base class for Tensorflow Training
    """

    def __init__(self, session, model, dataset, config):
        """
        Initializer fot BaseTraing object

        :param tf.Session session: tensorflow session
        :param BaseModel model: tensorflow model
        :param BaseDataset dataset: dataset object
        :param ConfigReader config: config reader object
        """
        self.session = session
        self.model = model
        self.dataset = dataset
        self.config = config


    def train(self):
        """
        Main training method.
        It creates tf.Dataset iterator from the Dataset and builds the tensorflow model.
        It runs the training epoches while logging the progress to Tensorboard.
        It has the capabilities to restore and save trained models.
        """
        raise NotImplementedError


    def dataset_iterator(self, mode='train'):
        """
        Method to be overridden generating tf.Dataset iterator
        :param str mode: idicator for 'train' or 'test'
        :return tuple(dict, tf.Iterator)
        """
        raise NotImplementedError

    def train_epoch(self, cur_epoche):
        """
        Method to be overridden for training epoche.
        :param int cur_epoche: index of current epoch
        """
        raise NotImplementedError

    def train_step(self):
        """
        Method to be overridden for training step.
        """
        raise NotImplementedError

    def test_step(self):
        """
        Method to be overridden for training step.
        """
        raise NotImplementedError

    def log_progress(self, input, num_iteration, mode):
        """
        Method to be overridden for logging the training progress to Tensorboard
        """
        raise NotImplementedError

    def update_progress_bar(self, t_bar, train_output, test_output):
        """
        Method to be overridden for updating tqdm progress bar
        """
        raise NotImplementedError
