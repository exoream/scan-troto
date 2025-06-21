import tensorflow as tf
import os

def load_model():
    model_path = os.path.abspath("model/model.h5")
    model = tf.keras.models.load_model(model_path)
    return model
