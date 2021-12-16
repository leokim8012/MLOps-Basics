import tensorflow as tf

from layers import losses
from models import model
from trainers import trainer


import numpy as np
from matplotlib import pyplot as plt


class ModelTrainer(trainer.Trainer):

    def __init__(
            self,
            model_parameters,
            model: model.Model,
            model_optimizer,
            callbacks=None,
    ):
        self.model = model
        self.model_optimizer = model_optimizer

        self.latent_size = model_parameters.latent_size
        super().__init__(
            model_parameters=model_parameters,
            models={'model': model},
            models_optimizers={
                'model_optimizer': self.model_optimizer
            },
            callbacks=callbacks,
        )

    @tf.function
    def train_step(self, batch):
        real_examples = batch
        with tf.GradientTape(persistent=True) as tape:

            real_output = self.model(real_examples, training=True)

            model_loss = losses.discriminator_loss(real_output)

        gradients_of_model = tape.gradient(
            target=model_loss,
            sources=self.model.trainable_variables,
        )

        self.model_optimizer.apply_gradients(
            grads_and_vars=zip(gradients_of_model, self.model.trainable_variables)
        )

        return {
            'model_loss': model_loss
        }
