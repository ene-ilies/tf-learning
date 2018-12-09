import tensorflow as tf

def noConvModel():
	model = tf.keras.models.Sequential([
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(512, activation=tf.nn.relu),
		tf.keras.layers.Dropout(0.2),
 		tf.keras.layers.Dense(10, activation=tf.nn.softmax)
	])
	model.compile(optimizer='adam',
 		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])
	return model


def oneConvModel():
	model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation=tf.nn.relu),
	        tf.keras.layers.MaxPool2D(),
	        tf.keras.layers.Flatten(),
	        tf.keras.layers.Dense(512, activation=tf.nn.relu),
	        tf.keras.layers.Dropout(0.2),
	        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
	])
	adam = tf.keras.optimizers.Adam(0.001)
	model.compile(optimizer=adam,
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])
	return model
