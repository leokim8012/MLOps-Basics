import tensorflow as tf

from datasets.flower_photos import FlowerPhotosDataset as Dataset
from models.flower_classifier import FlowerClassifierModel as Model
from trainers.trainer import Trainer as Trainer

from utils import config

def main():
  problem_params = config.read_config('flower_photos')


  dataset = Dataset(problem_params)
  model = Model(problem_params)



  checkpoint_filepath = '/tmp/checkpoint'

  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_accuracy',
      mode='max',
      save_best_only=True)

  # early_stopping_callback = EarlyStopping(
  #     monitor="val_loss", patience=3, verbose=True, mode="min"
  # )

  tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True,
    write_images=False, write_steps_per_second=False, update_freq='epoch',
    profile_batch=0, embeddings_freq=0, embeddings_metadata=None
  )



  trainer = Trainer(
    model_parameters = problem_params,
    model=model,
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    callbacks=[checkpoint_callback, tensorboard_callback]
  )

  trainer.compile()
  trainer.train(dataset=dataset)




if __name__ == '__main__':
    main()