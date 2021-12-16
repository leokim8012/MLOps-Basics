import abc
from abc import abstractmethod

class Dataset(abc.ABC):

    def __init__(self, input_params, with_labels=False,):
        self.batch_size = input_params.batch_size
        self.buffer_size = input_params.buffer_size
        self.img_height = input_params.img_height
        self.img_width = input_params.img_width
        self.data_dir = ''
        self.prepare_data()

        self.train_dataset = self.train_dataloader()
        self.val_dataset = self.val_dataloader()

    @abstractmethod
    def prepare_data(self):
        raise NotImplementedError

    @abstractmethod
    def train_dataloader(self):
        raise NotImplementedError
    @abstractmethod
    def val_dataloader(self):
        raise NotImplementedError

    def __iter__(self):
        return iter(self.train_dataset)
