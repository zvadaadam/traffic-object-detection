from object_detection.trainer.base_trainer import BaseTrain
from object_detection.dataset.image_iterator import ImageIterator
from object_detection.utils.tensor_logger import TensorLogger


class Trainer(BaseTrain):

    def __init__(self, session, model, dataset, config):
        super(Trainer, self).__init__(session, model, dataset, config)

        self.logger = TensorLogger(log_path=self.config.tensorboard_path(), session=self.session)
        self.iterator = ImageIterator(self.dataset, self.model, self.session, self.config)

    def dataset_iterator(self, mode='train'):

        # model_train_inputs, train_handle = self.iterator.create_dataset_iterator(mode='train')
        # _, test_handle = self.iterator.create_dataset_iterator(mode='test')

        model_train_inputs, train_handle = self.iterator.generating_iterator()

        return model_train_inputs, train_handle

    def train_epoch(self, cur_epoche):

        num_iterations = self.config.num_iterations()

        mean_loss = 0
        mean_acc = 0
        for i in range(num_iterations):

            loss, acc = self.train_step()

            mean_loss += loss
            mean_acc += acc

        mean_loss /= num_iterations
        mean_acc /= num_iterations

        return mean_loss, mean_acc


    def train_step(self):

        _, loss, acc = self.session.run([self.model.opt, self.model.loss, self.model.acc], feed_dict={
                            self.iterator.handle_placeholder: self.train_handle
                      })

        self.session.run(self.model.increment_global_step_tensor)

        return loss, acc

    def test_step(self):

        loss, acc = self.session.run([self.model.loss, self.model.acc], feed_dict={
                            self.iterator.handle_placeholder: self.test_handle
                      })

        return loss, acc


    def log_progress(self, input, num_iteration, mode):

        summaries_dict = {
            'loss': input[0],
            'acc': input[1],
        }

        self.logger.log_scalars(num_iteration, summarizer=mode, summaries_dict=summaries_dict)

    def update_progress_bar(self, t_bar, train_output, test_output):

        t_bar.set_postfix(
            train_loss='{:05.3f}'.format(train_output[0]),
            train_acc='{:05.3f}'.format(train_output[1]),
            test_loss='{:05.3f}'.format(test_output[0]),
            test_acc='{:05.3f}'.format(test_output[1]),
        )
