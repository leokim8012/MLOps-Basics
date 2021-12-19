from easydict import EasyDict as edict

from tensorflow.python.keras import Input
from tensorflow.python.keras import Model
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import layers


from models import model
class FlowerClassifierModel(model.Model):
  def __init__(self, model_parameters: edict):
    super().__init__(model_parameters)
  

  def define_model(self):
    # input_img = Input(shape=[
    #     self.model_parameters.img_height,
    #     self.model_parameters.img_width,
    #     self.model_parameters.num_channels,
    # ])
    model = Sequential([
      layers.Rescaling(1./255, input_shape=(self.model_parameters.img_height, self.model_parameters.img_width, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(self.model_parameters.num_channels)
    ])

    return model

  