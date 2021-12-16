import os
import tensorflow as tf


from abc import abstractmethod
from typing import List

from tqdm import tqdm

from callbacks import basic_callbacks
from callbacks import callback

from datasets import abstract_dataset
from models import model


from tensorflow.keras.optimizers import Optimizer as optimizers


class Trainer:
    def __init__(
            self,
            model_parameters,
            
            models: List[model.Model],
            models_optimizers: List[optimizers],

            callbacks: List[callback.Callback] = None,
    ):
        self.model_parameters = model_parameters
        self.batch_size = model_parameters.batch_size
        self.models = models

        self.global_step = 0
        self.epoch = 0
        self.models_optimizers = models_optimizers

        self.save_images_every_n_steps = model_parameters.save_images_every_n_steps


        default_callbacks = [
            basic_callbacks.GlobalStepIncrementer(),
        ]
        self.callbacks = default_callbacks

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError

    def train(self, dataset: abstract_dataset.Dataset, num_epochs: int,):
        global_step = 0
        epoch_tqdm = tqdm(iterable=range(num_epochs),desc="Epochs")

        losses = []
        step_loss =[]

        for self.epoch in epoch_tqdm:
            self.on_epoch_begin()

            dataset_tqdm = tqdm(iterable=dataset,desc="Batches",leave=True)
            for batch in dataset_tqdm:
                self.on_training_step_begin()
                losses = self.train_step(batch)
                self.on_training_step_end()     

                if(self.global_step % self.save_images_every_n_steps == 0):

                    losses.append(losses['model_loss'].numpy())
                    step_loss.append(self.global_step)

                postfix = 'Step: ' + str(self.global_step) + ' | model Loss: ' + str(losses['model_loss'].numpy())
                dataset_tqdm.set_postfix_str(postfix)
                dataset_tqdm.refresh()

            self.on_epoch_end()

    def on_epoch_begin(self):
        for c in self.callbacks:
            c.on_epoch_begin(self)

    def on_epoch_end(self):
        for c in self.callbacks:
            c.on_epoch_end(self)

    def on_training_step_begin(self):
        for c in self.callbacks:
            c.on_training_step_begin(self)

    def on_training_step_end(self):
        for c in self.callbacks:
            c.on_training_step_end(self)


    def save_model(self, name):
        path=os.getcwd() + '/pretrained-models/' 
        self.generator.model.save(path + name + '.h5')