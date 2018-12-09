import tensorflow as tf
import models

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("No convolution model:")
print("Input shape: %s" % tf.shape(x_train))
model = models.noConvModel()
checkpoint_path = "no_conv/cp-{epoch:04d}.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

#model.fit(x_train, y_train, epochs=5, callbacks=[cp_callback])

print("Evaluating...")
print("%s" % model.evaluate(x_test, y_test))

print("One convolution model:")
x_train, x_test = x_train.reshape(x_train.shape[0], 28, 28, 1), x_test.reshape(x_test.shape[0], 28, 28, 1)
print("Input shape: %s" % tf.shape(x_train))

model = models.oneConvModel()
checkpoint_path = "one_conv/cp-{epoch:04d}.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

model.fit(x_train, y_train, epochs=10, callbacks=[cp_callback])
model.summary()
print("Evaluating...")
print("%s" % model.evaluate(x_test, y_test))

