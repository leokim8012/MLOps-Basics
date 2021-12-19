import os
import tensorflow as tf


from abc import abstractmethod
from typing import List

from tqdm import tqdm

from callbacks import callback

from datasets import abstract_dataset
from models import model


from tensorflow.keras.optimizers import Optimizer as optimizers
from tensorflow.keras.losses import Loss as loss


class Trainer:
    def __init__(
            self,
            model_parameters,
            
            model: model.Model,
            optimizer: optimizers,
            loss: loss,
            callbacks: List[callback.Callback] = None,
    ):
        self.model_parameters = model_parameters
        self.batch_size = model_parameters.batch_size
        self.model = model

        self.global_step = 0
        self.epoch = 0

        self.optimizer = optimizer
        self.loss = loss
        self.callbacks = callbacks


    def compile(self,):
        self.model.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['accuracy']
        )

        self.model.model.summary()

    def train(self, dataset: abstract_dataset.Dataset):

        model = self.model.model
        history = model.fit(
            dataset.train_dataset,
            validation_data=dataset.val_dataset,
            epochs=self.model_parameters.num_epochs,
            callbacks=self.callbacks
        )

    def save_model(self, name):
        path=os.getcwd() + '/pretrained-models/' 
        self.generator.model.save(path + name + '.h5')