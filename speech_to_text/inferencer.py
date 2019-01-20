import numpy as np
import tensorflow as tf
import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def inputlayer():
    audio = tf.placeholder(tf.float32, shape=(1, 16000), name="audio_input")

    spectrogram = tf.abs(tf.contrib.signal.stft(
            audio, frame_length=512, frame_step=243, fft_length=512))

    #spectrogram = tf.real(spectrogram * tf.conj(spectrogram))

    print_spectrogram = tf.print("spectrogram: ", spectrogram.shape)

    # Warp the linear-scale, magnitude spectrograms into the mel-scale.
    num_spectrogram_bins = spectrogram.shape[-1].value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0, 8000.0, 128
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, 16000, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
      spectrogram, linear_to_mel_weight_matrix, 1)

    mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

    print_mel_spectrogram = tf.print("mel spectrogram: ", mel_spectrograms.shape)

    brightness_placeholder = tf.constant(60., name="brightness_placeholder")

    mul = tf.multiply(mel_spectrograms, brightness_placeholder)

    min_const = tf.constant(255., name="min_const")

    minimum = tf.minimum(mel_spectrograms, min_const, name="min")

    #cast = tf.cast(minimum, tf.uint8, name="cast")

    #print_cast = tf.print("casted: ", cast)

    expand_dims = tf.expand_dims(minimum, -1)

    squeeze = tf.squeeze(expand_dims, 0)

    transposed = tf.transpose(squeeze, [1, 0, 2])

    reshaped = tf.reshape(transposed, (1, 128, 64, 1))

    return reshaped

class SpeechToTextInferencer:

    def __init__(self, checkpointDir, labels):
        weights = tf.train.latest_checkpoint(checkpointDir)

#        self.model = models.oneConvModelWithTFInput(inputlayer())
        self.model = models.oneConvModel()
        self.model.load_weights(weights)

        self.labels = labels

    def infertf(self, data):
        #with tf.Session() as sess:
        #    #tf.keras.backend.set_session(sess)
        #    #sess.run(tf.global_variables_initializer())
        #    softmax = self.model.output
        #    predictions = sess.run(softmax, {"audio_input:0": data})
        #    return self.labels[np.argmax(predictions)]
        predictions = self.model.predict({"audio": data})
        return self.labels[np.argmax(predictions)]

    def infer(self, data):
        reshaped = data.reshape((1, 128, 64, 1))
        predictions = self.model.predict({"conv2d_input": reshaped})
        return self.labels[np.argmax(predictions)]

    def savemodel(self, outputdir):
        with tf.keras.backend.get_session() as sess:
            writer = tf.summary.FileWriter(outputdir, sess.graph)
        writer.close()
        
