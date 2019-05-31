import sys
sys.path.append('/home/zvadaada/traffic-object-detection/')

import tensorflow as tf
import os
from tensorflow.python import debug as tf_debug
from object_detection.config.config_reader import ConfigReader
from object_detection.dataset.udacity_object_dataset import UdacityObjectDataset
from object_detection.model.YOLO import YOLO
from object_detection.trainer.object_trainer import ObjectTrainer
from tensorflow.python.client import timeline


def main_train(config: ConfigReader):

    dataset = UdacityObjectDataset(config)

    model = YOLO(config)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    #tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4

    with tf.Session(config=tf_config) as session:

        #session = tf_debug.TensorBoardDebugWrapperSession(session, 'PRGA-004810.local:6064', send_traceback_and_source_code=False)

        # add additional options to trace the session execution
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        trainer = ObjectTrainer(session, model, dataset, config, options, run_metadata)

        trainer.train()

        # Create the Timeline object, and write it to a json file
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_01.json', 'w') as f:
            f.write(chrome_trace)


if __name__ == '__main__':

    config_path = '/home/zvadaada/traffic-object-detection/config/test.yml'
    # config_path = '/Users/adam.zvada/Documents/Dev/object-detection/config/test.yml'
    # config_path = '/home/adam.zvada/dev/object-detection/config/test.yml'

    main_train(ConfigReader(config_path))
