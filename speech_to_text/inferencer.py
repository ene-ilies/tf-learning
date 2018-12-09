import numpy as np
import tensorflow as tf
import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class SpeechToTextInferencer:

    def __init__(self):
        checkpoint_dir = "no-unknown"
        weights = tf.train.latest_checkpoint(checkpoint_dir)

        self.model = models.oneConvModel()
        self.model.load_weights(weights)

        self.dataGenerator = ImageDataGenerator(1./255)
        self.labels = ['no', 'silence', 'yes']

    def infer(self, data):
        image = data.reshape((1, 128, 64, 1))
        predictions = self.model.predict_generator(self.dataGenerator.flow(image), steps=1)
        return self.labels[np.argmax(predictions)]
