import tensorflow as tf
mnist = tf.keras.datasets.mnist

_10sec = False  # Flag to keep operation under 10 seconds

# Hyperparameters
epochs = 5
dropout = 0.2
learning_rate = 0.001
hidden_layer_count = 1
hidden_layer_size = 128

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

if _10sec:
    epochs = 1
    x_train = x_train[:100]
    y_train = y_train[:100]
    x_test = x_test[:10]
    y_test = y_test[:10]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
for _ in range(hidden_layer_count):
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=epochs)
model.evaluate(x_test, y_test)