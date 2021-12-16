import tensorflow as tf

from datasets.fashion_mnist import FashionMnistDataset as Dataset
from models.discriminator import Discriminator as Model

from trainers.model_trainer import ModelTrainer

from utils import config

def main():
  problem_params = config.read_config('fashion_mnist')

  dataset = Dataset(problem_params).load_data()
  train_dataset = dataset

  model = Model(problem_params)

  trainer = ModelTrainer(
    model_parameters=problem_params,
    model=model, 
    model_optimizer=tf.keras.optimizers.Adam(
      learning_rate=problem_params.learning_rate_discriminator,
      beta_1=0.5,
    )
  )

  trainer.train(train_dataset, problem_params.num_epochs)


if __name__ == '__main__':
    main()