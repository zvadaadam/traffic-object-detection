from object_detection.model.cnn_model import CNNModel
from object_detection.config.config_reader import ConfigReader


class Resnet50(CNNModel):
    """
    ResNet50 architecture
    """
    def __init__(self, config: ConfigReader):
        super(Resnet50, self).__init__(config)

    def build_network(self, x, y):
        # TODO: Resent50
        pass







