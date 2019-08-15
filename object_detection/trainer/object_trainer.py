import tensorflow as tf
import numpy as np
from tqdm import trange
from object_detection.trainer.base_trainer import BaseTrain
from object_detection.dataset.image_iterator import ImageIterator
from object_detection.utils.tensor_logger import TensorLogger
from object_detection.config.config_reader import ConfigReader
from object_detection.utils import image_utils
from object_detection.utils import yolo_utils


class ObjectTrainer(BaseTrain):

    def __init__(self, session: tf.Session(), model, dataset, config: ConfigReader, options, run_metadata):
        super(ObjectTrainer, self).__init__(session, model, dataset, config)

        self.options = options
        self.run_metadata = run_metadata

        self.iterator = ImageIterator(self.session, self.dataset, self.config, self.model)

        self.train_handle = None
        self.test_handle = None

    def dataset_iterator(self, mode='train'):

        print('Init iterators...')

        x, y, train_handle = self.iterator.create_iterator(mode='train')
        _, _, test_handle = self.iterator.create_iterator(mode='test')

        print('Done creating iterators...')

        # x, y, train_handle = self.iterator.create_iterator_from_tfrecords(mode='train')
        # _, test_handle = self.iterator.create_iterator_from_tfrecords(mode='test')

        return x, y, train_handle, test_handle

    def train(self):
        """
        Main training method.
        It creates tf.Dataset iterator from the Dataset and builds the tensorflow model.
        It runs the training epoches while logging the progress to Tensorboard.
        It has the capabilities to restore and save trained models.
        """

        x, y, train_handle, test_handle = self.dataset_iterator()

        self.train_handle = train_handle
        self.test_handle = test_handle

        self.model.build_model(mode='train', x=x, y=y)

        train_writer, test_writer, merged_summaries = self.init_tensorboard()

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(init)

        self.model.init_saver(max_to_keep=2)

        # restore latest checkpoint model
        if self.config.restore_trained_model() is not None:
            self.model.load(self.session, self.config.restore_trained_model())

        # tqdm progress bar looping through all epoches
        t_epoches = trange(self.model.cur_epoch_tensor.eval(self.session), self.config.num_epoches() + 1, 1,
                           desc='Training {}'.format(self.config.model_name()))
        for cur_epoch in t_epoches:
            # run epoch training
            train_output = self.train_epoch(cur_epoch, train_writer, merged_summaries)
            # run model on test set
            test_output = self.test_step(test_writer, cur_epoch, merged_summaries)

            self.update_progress_bar(t_epoches, train_output, test_output)

            # increase epoche counter
            self.session.run(self.model.increment_cur_epoch_tensor)

            if cur_epoch % 10 == 0:
                self.model.save(self.session, write_meta_graph=True)

        # finale save model - creates checkpoint
        self.model.save(self.session, write_meta_graph=True)

    def eval(self, num_iterations=1):

        x, y, train_handle, test_handle = self.dataset_iterator()

        self.test_handle = test_handle
        self.train_handle = train_handle

        self.model.build_model(mode='train', x=x, y=y)

        train_writer, test_writer, merged_summaries = self.init_tensorboard()

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(init)

        self.model.init_saver(max_to_keep=2)

        # restore latest checkpoint model
        if self.config.restore_trained_model() is not None:
            self.model.load(self.session, self.config.restore_trained_model())

        for i in range(num_iterations):
            # run model on test set
            test_output = self.test_step(test_writer, i, merged_summaries)


    def init_tensorboard(self):

        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(self.config.tensorboard_path() + '/train', self.session.graph)
        test_writer = tf.summary.FileWriter(self.config.tensorboard_path() + '/test', self.session.graph)

        return train_writer, test_writer, merged

    def train_epoch(self, cur_epoche, train_writer, merged_summaries):

        num_iterations = self.config.num_iterations()

        mean_loss = 0
        for i in range(num_iterations):

            if i % 100 == 0:
                loss = self.train_step(train_writer, cur_epoche * num_iterations + i, merged_summaries)
            else:
                loss = self.train_step(train_writer, cur_epoche * num_iterations + i)

            mean_loss += loss

        mean_loss /= num_iterations

        return mean_loss


    def train_step(self, train_writer, num_iter, merged_summaries=None):

        # run training
        if merged_summaries != None:
            loss, _, summary = self.session.run([self.model.loss, self.model.opt, merged_summaries],
                                                feed_dict={self.iterator.handle_placeholder: self.train_handle,
                                                           self.model.is_training: True},
                                                options=self.options, run_metadata=self.run_metadata)

            # write summaries to tensorboard
            train_writer.add_summary(summary, num_iter)
        else:
            loss, _ = self.session.run([self.model.loss, self.model.opt],
                                       feed_dict={self.iterator.handle_placeholder: self.train_handle,
                                                  self.model.is_training: True})

        print(loss)

        # increase global step
        self.session.run(self.model.increment_global_step_tensor)

        return loss

    def test_step(self, test_writer, cur_epoche, merged_summaries):

        num_iterations = self.config.num_iterations() * cur_epoche

        loss, summary, loss_cord, loss_size, loss_obj, loss_noobj, loss_class, learning_rate, nan_1, nan_2, nan_3 = self.session.run(
            [self.model.loss,
             merged_summaries,
             self.model.loss_cord,
             self.model.loss_size,
             self.model.loss_obj,
             self.model.loss_noobj,
             self.model.loss_class,
             self.model.learning_rate,
             self.model.nan_1, self.model.nan_2, self.model.nan_3],
            feed_dict={self.iterator.handle_placeholder: self.train_handle,
                       self.model.is_training: True})

        #test_writer.add_summary(summary, num_iterations)

        print(np.any(nan_1))
        print(np.any(nan_2))
        print(np.any(nan_3))

        print(f'Loss: {loss} -  cord: {loss_cord}, size: {loss_size}, obj: {loss_obj}, noobj: {loss_noobj}, class: {loss_class}')

        # -----------TESTING-----------
        # np.set_printoptions(formatter={'float_kind': '{:f}'.format})
        #
        # (boxes, scores, classes) = output
        #
        # if boxes.shape[0] == 0:
        #     print('NO BOXES RETURNED!')
        #     return loss, loss_cord, loss_size, loss_obj, loss_noobj, loss_class, learning_rate
        #
        # print(f'Predicted: {boxes[0][0]}, {boxes[0][1]}, {boxes[0][2]}, {boxes[0][3]}')
        #
        # label_boxes = []
        # for label in np.reshape(input['y'][0], newshape=[49, 3, 10]):
        #     label_boxes.append(label)
        #
        # for img in input['x']:
        #     image = img / 255
        #     for box in output[0]:
        #         image = image_utils.add_bb_to_img(image, box[0], box[1], box[2], box[3])
        #
        #     image_utils.plot_img(image)
        # -----------TESTING-----------

        return loss, loss_cord, loss_size, loss_obj, loss_noobj, loss_class, learning_rate

    def update_progress_bar(self, t_bar, train_output, test_output):

        t_bar.set_postfix(
            train_loss='{:05.3f}'.format(train_output),
            test_loss='{:05.3f}'.format(test_output[0]),
            test_loss_cord='{:05.3f}'.format(test_output[1]),
            test_loss_size='{:05.3f}'.format(test_output[2]),
            test_loss_obj='{:05.3f}'.format(test_output[3]),
            test_loss_noobj='{:05.3f}'.format(test_output[4]),
            test_loss_class='{:05.3f}'.format(test_output[5]),
            test_learning_rate='{:05.5f}'.format(test_output[6]),
        )
