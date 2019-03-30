import pandas as pd
from sklearn.model_selection import train_test_split
from object_detection.config.config_reader import ConfigReader

class DatasetBase(object):
    """
    Base class for datasets operation.
    """

    def __init__(self, config: ConfigReader):
        self.config = config

        self.df = pd.DataFrame()
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.validate_df = pd.DataFrame()

    def load_dataset(self):
        raise NotImplemented

    def split_dataset(self, test_size=None):

        if test_size is None:
            test_size = self.config.test_size()

        train, test = train_test_split(self.df, test_size=test_size, random_state=42, shuffle=False)

        self.train_df = train
        self.test_df = test

    def set_train_df(self, df):
        self.train_df = df

    def set_test_df(self, df):
        self.test_df = df

    def train_dataset(self):
        return self.train_df

    def test_dataset(self):
        return self.test_df

    def validate_dataset(self):
        return self.validate_df

