import os
import tensorflow as tf
from object_detection.config.config_reader import ConfigReader
from object_detection.dataset.udacity_object_dataset import UdacityObjectDataset
from object_detection.model.YOLO import YOLO
from object_detection.trainer.object_trainer import ObjectTrainer



def main_train(config: ConfigReader):

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    dataset = UdacityObjectDataset(config)

    model = YOLO(config)

    with tf.Session() as session:
        trainer = ObjectTrainer(session, model, dataset, config)

        trainer.train()


if __name__ == '__main__':

    config_path = '/Users/adam.zvada/Documents/Dev/object-detection/config/test.yml'

    main_train(ConfigReader(config_path))
