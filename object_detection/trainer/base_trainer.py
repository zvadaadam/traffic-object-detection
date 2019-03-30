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

        model_train_inputs, train_handle, test_handle = self.dataset_iterator()

        self.train_handle = train_handle
        self.test_handle = test_handle

        self.model.build_model(model_train_inputs)

        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(self.init)

        self.model.init_saver(max_to_keep=2)

        # restore latest checkpoint model
        if self.config.restore_trained_model() != None:
            self.model.load(self.session, self.config.restore_trained_model())

        # tqdm progress bar looping through all epoches
        t_epoches = trange(self.model.cur_epoch_tensor.eval(self.session), self.config.num_epoches() + 1, 1,
                           desc='Training {}'.format(self.config.model_name()))
        for cur_epoch in t_epoches:

            # run epoch training
            train_output = self.train_epoch(cur_epoch)
            # run model on test set
            test_output = self.test_step()

            self.log_progress(train_output, num_iteration=cur_epoch * self.config.num_iterations(), mode='train')
            self.log_progress(test_output, num_iteration=cur_epoch * self.config.num_iterations(), mode='test')

            self.update_progress_bar(t_epoches, train_output, test_output)

            # increase epoche counter
            self.session.run(self.model.increment_cur_epoch_tensor)

            self.model.save(self.session, write_meta_graph=False)

        # finale save model - creates checkpoint
        self.model.save(self.session, write_meta_graph=True)

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
