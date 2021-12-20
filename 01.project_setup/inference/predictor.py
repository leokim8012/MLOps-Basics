import tensorflow as tf

from datasets.flower_photos import FlowerPhotosDataset as Dataset
from models.flower_classifier import FlowerClassifierModel as Model
from utils import config

class Predictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = tf.keras.models.load_model(self.model_path)


    def predict(self, img):
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        return predictions
