import tensorflow as tf

def oneConvModel():
	model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(input_shape=(128, 64, 1), filters=64, kernel_size=(8, 20), activation=tf.nn.relu),
	        tf.keras.layers.MaxPool2D(),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 10), activation=tf.nn.relu),
	        tf.keras.layers.Flatten(),
		tf.keras.layers.Dropout(0.1),
	        tf.keras.layers.Dense(7, activation=tf.nn.softmax)
	])
	adam = tf.keras.optimizers.Adam(0.0001)
	model.compile(optimizer=adam,
		loss='categorical_crossentropy',
		metrics=['accuracy'])
	return model

def oneConvModelWithTFInput(tfInput):
	model = tf.keras.models.Sequential([
		tf.keras.layers.InputLayer(input_tensor=tfInput),
		tf.keras.layers.Conv2D(filters=64, kernel_size=(8, 20), activation=tf.nn.relu),
	        tf.keras.layers.MaxPool2D(),
		tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 10), activation=tf.nn.relu),
	        tf.keras.layers.Flatten(),
	        tf.keras.layers.Dense(7, activation=tf.nn.softmax, name="softmax")
	])
	adam = tf.keras.optimizers.Adam(0.0001)
	model.compile(optimizer=adam,
		loss='categorical_crossentropy',
		metrics=['accuracy'])
	return model
