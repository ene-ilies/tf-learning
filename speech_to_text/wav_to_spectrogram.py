import argparse
import librosa
from PIL import Image
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '-f', '--wavfile',
    help='file to convert to spectrogram')
parser.add_argument(
    '-o', '--outputfile',
    help='file to convert to spectrogram')
args = parser.parse_args()

def tfspectrogram(inputfile, outputfile):
    audio = tf.read_file(inputfile, name="input_wav")
    my_audio = tf.contrib.ffmpeg.decode_audio(audio, "wav", 16000, 1)

    reshaped_audio = tf.reshape(my_audio, [1, -1])
    print_reshaped = tf.print("reshaped: ", reshaped_audio.shape)

    with tf.control_dependencies([print_reshaped]):
        spectrogram = tf.abs(tf.contrib.signal.stft(
            reshaped_audio, frame_length=512, frame_step=243, fft_length=512))

    spectrogram = tf.real(spectrogram * tf.conj(spectrogram))

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

    brightness_placeholder = tf.placeholder(tf.float32, name="brightness_placeholder")

    mul = tf.multiply(mel_spectrograms, brightness_placeholder)

    min_const = tf.constant(255., name="min_const")

    minimum = tf.minimum(mel_spectrograms, min_const, name="min")

    cast = tf.cast(minimum, tf.uint8, name="cast")

    print_cast = tf.print("casted: ", cast)

    expand_dims = tf.expand_dims(cast, -1)

    squeeze = tf.squeeze(expand_dims, 0)

    transposed = tf.transpose(squeeze, [1, 0, 2])

    png_encoder = tf.image.encode_png(transposed, name="png_encoder")

    with tf.control_dependencies([print_cast, print_spectrogram, print_mel_spectrogram]):
        file_writer = tf.write_file(outputfile, png_encoder, name="file_writer")

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("7-labels/logs/", sess.graph)
        sess.run(file_writer, feed_dict={brightness_placeholder: 60.})

    writer.close()

def librosaspectrogram(inputfile, outputfile):
    data, sampleRate = librosa.load(inputfile, sr=None)
    melSpectrogram = librosa.feature.melspectrogram(y=data, sr=sampleRate, n_mels=128, hop_length=252, power=2.0)
    melSpectrogram = melSpectrogram.astype(np.uint8)
    Image.fromarray(melSpectrogram).save(outputfile)

tfspectrogram(args.wavfile, args.outputfile)
librosaspectrogram(args.wavfile, args.outputfile.replace(".png", "-librosa.png"))





