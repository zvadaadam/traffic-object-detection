import os
import pandas as pd
import numpy as np
from object_detection.config.config_reader import ConfigReader
from object_detection.dataset.dataset_base import DatasetBase
from object_detection.dataset.udacity_object_dataset import UdacityObjectDataset
from object_detection.dataset.rovit_dataset import RovitDataset

class AllDataset(DatasetBase):

    def __init__(self, config: ConfigReader, dataset_names=None):
        super(AllDataset, self).__init__(config)

        # TODO: load anchores from config!

        # TODO: hook up dataset_names
        # TODO: add factory pattern from datasets
        self.udacity_dataset = UdacityObjectDataset(config)
        self.rovit_dataset = RovitDataset(config)

    def load_annotation_df(self):

        udacity_annotation_df = self.udacity_dataset.load_annotation_df()
        rovit_dataset = self.rovit_dataset.load_annotation_df()

        annotation_df = udacity_annotation_df.append(rovit_dataset)

        annotation_df.to_csv('alldatset.csv')

        return annotation_df


if __name__ == '__main__':

    from object_detection.config.config_reader import ConfigReader

    config = ConfigReader()

    dataset = AllDataset(config)
    dataset.load_dataset()
