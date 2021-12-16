import tensorflow as tf

from datasets.flower_photos import FlowerPhotosDataset as Dataset
from models.flower_classifier import FlowerClassifierModel as Model

from utils import config

def main():
  problem_params = config.read_config('flower_photos')

  dataset = Dataset(problem_params)
  train_dataset = dataset

  print(train_dataset.train_dataset)

  model = Model(problem_params)
  model.model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

  model.model.summary()


if __name__ == '__main__':
    main()