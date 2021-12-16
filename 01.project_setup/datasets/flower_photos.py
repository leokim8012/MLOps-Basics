import tensorflow as tf
from utils import data_utils
import pathlib

from datasets import abstract_dataset
class FlowerPhotosDataset(abstract_dataset.Dataset):

    def __init__(
            self,
            input_params,
            with_labels=False,
    ):
        super().__init__(input_params, with_labels)

    def __call__(self, *args, **kwargs):
        return self.train_dataset


    def prepare_data(self):
        import os
        path=os.getcwd() + '/datasets/'
        self.data_dir = path+r'flower_photos'


        if(pathlib.Path(self.data_dir+'.tgz').is_file == False):
            import wget
            url='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
            wget.download(url)



        if(pathlib.Path(self.data_dir).is_dir() == False):
            data_utils.extract(path, extract_path='./datasets')

        self.data_dir = pathlib.Path(path)
        image_count = len(list(self.data_dir.glob('*/*.jpg')))


    def train_dataloader(self):

        # train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(
        #     self.buffer_size).batch(
        #     self.batch_size)
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )
        # train_ds = data_utils.normalize_inputs(train_ds)
        return train_ds

    def val_dataloader(self):
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )
        return val_ds