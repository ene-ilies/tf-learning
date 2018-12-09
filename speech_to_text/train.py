import tensorflow as tf
import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 16

dataGenerator = ImageDataGenerator(1./255)
trainDataGenerator = dataGenerator.flow_from_directory('images/train', target_size=(128, 64), color_mode='grayscale', batch_size=batch_size, shuffle=True, class_mode='categorical')
print("train: ", trainDataGenerator.class_indices)
validationDataGenerator = dataGenerator.flow_from_directory('images/test', target_size=(128, 64), color_mode='grayscale', batch_size=batch_size, shuffle=False, class_mode='categorical')
print("validation: ", validationDataGenerator.class_indices)

print("One convolution model:")
model = models.oneConvModel()
checkpoint_path = "no-unknown/cp-{epoch:04d}.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

# model.fit_generator(trainDataGenerator, steps_per_epoch=5000/batch_size, epochs=50, validation_data=validationDataGenerator, validation_steps=803/batch_size, callbacks=[cp_callback])

