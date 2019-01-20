import argparse
import os
import tensorflow as tf
import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '-t', '--train',
    help='Directory where train images are located each class in its directory.')
parser.add_argument(
    '-v', '--validation',
    help='Directory where validation images are located each class in its directory.')
parser.add_argument(
    '-cp', '--checkpointdir',
    help='Directory where to save checkpoint files.')
parser.add_argument(
    '-tb', '--tensorboardlogdir',
    help='Directory where to save tensorboard logs.')
args = parser.parse_args()

batch_size = 16
checkpoint_path = os.path.join(args.checkpointdir, "cp-{epoch:04d}.ckpt") 
tensorboard_log_dir = os.path.join(args.tensorboardlogdir, "")

train_dir = os.path.join(args.train, "")
validation_dir = os.path.join(args.validation, "")

print("train %s" % train_dir)
print("validation %s" % validation_dir)
print("checkpoint_path %s" % checkpoint_path)
print("tensorboard_log_dir %s" % tensorboard_log_dir)

dataGenerator = ImageDataGenerator(1./255)
trainDataGenerator = dataGenerator.flow_from_directory(train_dir, target_size=(128, 64), color_mode='grayscale', batch_size=batch_size, shuffle=True, class_mode='categorical')
print("train: ", trainDataGenerator.class_indices)
validationDataGenerator = dataGenerator.flow_from_directory(validation_dir, target_size=(128, 64), color_mode='grayscale', batch_size=batch_size, shuffle=False, class_mode='categorical')
print("validation: ", validationDataGenerator.class_indices)

print("One convolution model:")
model = models.oneConvModel()

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=tensorboard_log_dir, histogram_freq=5, write_graph=True)

model.fit_generator(trainDataGenerator, steps_per_epoch=5000/batch_size, epochs=50, validation_data=validationDataGenerator, validation_steps=803/batch_size, 
    callbacks=[cp_callback, tensorboard_callback])

