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

    def __init__(self, session: tf.Session(), model, dataset, config: ConfigReader):
        super(ObjectTrainer, self).__init__(session, model, dataset, config)

        self.logger = TensorLogger(log_path=self.config.tensorboard_path(), session=self.session)
        self.iterator = ImageIterator(self.session, self.model, self.dataset, self.config)

    def dataset_iterator(self, mode='train'):

        # model_train_inputs, train_handle = self.iterator.create_dataset_iterator(mode='train')
        # _, test_handle = self.iterator.create_dataset_iterator(mode='test')

        model_train_inputs, train_handle = self.iterator.create_iterator(mode='train')
        _, test_handle = self.iterator.create_iterator(mode='test')

        return model_train_inputs, train_handle, test_handle

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

        train_writer, test_writer, merged_summaries = self.init_tensorboard()

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
            train_output = self.train_epoch(cur_epoch, train_writer, merged_summaries)
            # run model on test set
            test_output = self.test_step(test_writer, merged_summaries)

            self.log_progress(train_output, num_iteration=cur_epoch * self.config.num_iterations(), mode='train')
            self.log_progress(test_output, num_iteration=cur_epoch * self.config.num_iterations(), mode='test')

            self.update_progress_bar(t_epoches, train_output, test_output)

            # increase epoche counter
            self.session.run(self.model.increment_cur_epoch_tensor)

            self.model.save(self.session, write_meta_graph=False)

        # finale save model - creates checkpoint
        self.model.save(self.session, write_meta_graph=True)

    def init_tensorboard(self):

        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(self.config.tensorboard_path() + '/train', self.session.graph)
        test_writer = tf.summary.FileWriter(self.config.tensorboard_path() + '/test')

        return train_writer, test_writer, merged

    def train_epoch(self, cur_epoche, train_writer, merged_summaries):

        num_iterations = self.config.num_iterations()

        mean_loss = 0
        for i in range(num_iterations):

            loss = self.train_step(train_writer, merged_summaries, cur_epoche*num_iterations + i)

            mean_loss += loss

        mean_loss /= num_iterations

        return mean_loss, 0


    def train_step(self, train_writer, merged_summaries, iter):

        # run training
        summary, _, loss = self.session.run([merged_summaries, self.model.opt, self.model.loss],
            feed_dict={self.iterator.handle_placeholder: self.train_handle}
        )

        # write summaries to tensorboard
        train_writer.add_summary(summary, iter)

        # increase global step
        self.session.run(self.model.increment_global_step_tensor)

        return loss

    def test_step(self, test_writer, merged_summaries):

        loss, loss_cord, loss_size, loss_obj, loss_noobj, loss_class, learning_rate, boxes, scores, classes, image, labels, mask, debug = self.session.run(
            [self.model.get_loss(),
             self.model.loss_cord,
             self.model.loss_size,
             self.model.loss_obj,
             self.model.loss_noobj,
             self.model.loss_class,
             self.model.learning_rate,
             self.model.get_tensor_boxes(),
             self.model.get_tensor_scores(),
             self.model.get_tensor_classes(),
             self.model.get_image(),
             self.model.get_labels(),
             self.model.mask,
             self.model.debug],
            feed_dict={self.iterator.handle_placeholder: self.test_handle}
        )
        np.set_printoptions(formatter={'float_kind': '{:f}'.format})
        print(debug)

        # -----------TESTING-----------
        if boxes.shape[0] == 0:
            print('NO BOXES RETURNED!')
            return loss, loss_cord, loss_size, loss_obj, loss_noobj, loss_class, learning_rate

        label_boxes = []
        for label in np.reshape(labels[0], newshape=[49, 3, 10]):
            label_boxes.append(label)

        print(f'Predicted: {boxes[0][0]}, {boxes[0][1]}, {boxes[0][2]}, {boxes[0][3]}')
        print(f'Label: {label_boxes[0][0][0]}, {label_boxes[0][0][1]}, {label_boxes[0][0][2]}, {label_boxes[0][0][3]}')

        #image_utils.plot_img(image_utils.add_bb_to_img(image[0], boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]))

        # x_min, y_min, x_max, y_max = yolo_utils.from_yolo_to_cord(boxes[0], image[0].shape)
        # print(f'Predicted: {x_min}, {y_min}, {x_max}, {y_max}')
        # image_utils.add_bb_to_img(image[0], x_min, y_min, x_max, y_max)
        #
        #
        # x_min, y_min, x_max, y_max = yolo_utils.from_yolo_to_cord(label_boxes[0][0:4], image[0].shape)
        # print(f'Label: {x_min}, {y_min}, {x_max}, {y_max}')
        # image_utils.plot_img(image_utils.add_bb_to_img(image[0], x_min, y_min, x_max, y_max))
        # -----------TESTING-----------

        return loss, loss_cord, loss_size, loss_obj, loss_noobj, loss_class, learning_rate

    def log_progress(self, input, num_iteration, mode):

        summaries_dict = {
            'loss': input[0]
        }

        self.logger.log_scalars(num_iteration, summarizer=mode, summaries_dict=summaries_dict)

    def update_progress_bar(self, t_bar, train_output, test_output):

        t_bar.set_postfix(
            train_loss='{:05.3f}'.format(train_output[0]),
            #train_acc='{:05.3f}'.format(train_output[1]),
            test_loss='{:05.3f}'.format(test_output[0]),
            test_loss_cord='{:05.3f}'.format(test_output[1]),
            test_loss_size='{:05.3f}'.format(test_output[2]),
            test_loss_obj='{:05.3f}'.format(test_output[3]),
            test_loss_noobj='{:05.3f}'.format(test_output[4]),
            test_loss_class='{:05.3f}'.format(test_output[5]),
            test_learning_rate='{:05.5f}'.format(test_output[6]),
            #test_acc='{:05.3f}'.format(test_output[1]),
        )
