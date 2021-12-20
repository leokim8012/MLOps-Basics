import tensorflow as tf

from datasets.flower_photos import FlowerPhotosDataset as Dataset
from models.flower_classifier import FlowerClassifierModel as Model
from trainers.trainer import Trainer as Trainer
from inference.predictor import Predictor as Predictor

from utils import config

def train():
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

  early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False
  )

  tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs', histogram_freq=0, write_graph=True,
    write_images=False, write_steps_per_second=False, update_freq='epoch',
    profile_batch=0, embeddings_freq=0, embeddings_metadata=None
  )



  trainer = Trainer(
    model_parameters = problem_params,
    model=model,
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback]
  )
  trainer.compile()
  trainer.train(dataset=dataset)
  trainer.save_model('model')

def inference():

    problem_params = config.read_config('flower_photos')
    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
    img = tf.keras.preprocessing.image.load_img(
        sunflower_path, target_size=(problem_params.img_height, problem_params.img_width)
    )
    predictor = Predictor("pretrained-models/model.h5")
    print(predictor.predict(img))



if __name__ == '__main__':
    # For Training
    train()
    # For checking
    inference()