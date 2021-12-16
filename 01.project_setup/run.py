import tensorflow as tf

from datasets.flower_photos import FlowerPhotosDataset as Dataset

from utils import config

def main():
  problem_params = config.read_config('flower_photos')

  dataset = Dataset(problem_params)
  train_dataset = dataset

  print(train_dataset.train_dataset)

  # model = Model(problem_params)


if __name__ == '__main__':
    main()