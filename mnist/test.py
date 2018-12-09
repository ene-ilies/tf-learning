import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import models

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = x_train.reshape(x_train.shape[0], 28, 28, 1), x_test.reshape(x_test.shape[0], 28, 28, 1)

checkpoint_dir = "one_conv"
weights = tf.train.latest_checkpoint(checkpoint_dir)

model = models.oneConvModel()
model.load_weights(weights)

print("Evaluating...")
print("%s" % model.evaluate(x_test, y_test))

print("Testing...")
randInt = np.random.randint(low=0, high=100)
plt.imshow(x_test[randInt])
plt.show()

predictions = model.predict(np.array([x_test[randInt]]))
predictedIndexes = [np.argmax(p) for p in predictions]
print("Result: %s" % predictedIndexes)

