import tensorflow as tf
from object_detection.trainer.base_trainer import BaseTrain
from object_detection.dataset.image_iterator import ImageIterator
from object_detection.utils.tensor_logger import TensorLogger
from object_detection.config.config_reader import ConfigReader


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

    def train_epoch(self, cur_epoche):

        num_iterations = self.config.num_iterations()

        mean_loss = 0
        for i in range(num_iterations):

            loss = self.train_step()

            mean_loss += loss

        mean_loss /= num_iterations

        return mean_loss, 0


    def train_step(self):

        _, loss = self.session.run([self.model.opt, self.model.loss],
            feed_dict={self.iterator.handle_placeholder: self.train_handle}
        )

        self.session.run(self.model.increment_global_step_tensor)

        return loss

    def test_step(self):

        loss, loss_cord, loss_confidence, loss_class, box = self.session.run(
            [self.model.loss, self.model.loss_cord, self.model.loss_confidence, self.model.loss_class, self.model.boxes],
            feed_dict={self.iterator.handle_placeholder: self.test_handle}
        )

        print(box)

        return loss, loss_cord, loss_confidence, loss_class


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
            test_loss_coenfience='{:05.3f}'.format(test_output[2]),
            test_loss_class='{:05.3f}'.format(test_output[3]),
            #test_acc='{:05.3f}'.format(test_output[1]),
        )
