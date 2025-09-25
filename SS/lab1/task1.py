from tabnanny import verbose

import tensorflow as tf
import numpy as np
import os

model_file = r"model_file.keras"

a = np.random.uniform(0, 10, size=10)
b = np.random.uniform(0, 10, size=10)
c = a + b

x= np.column_stack((a, b))
y = c



if os.path.exists(model_file):
    print("Loading model file...")
    model = tf.keras.models.load_model(model_file)
    print("Model is loaded!")
else:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.summary()

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    print("Learning...")
    history = model.fit(x, y, epochs=10000, verbose=False)

    print(f"Save model: {model_file}")
    model.save(model_file)
    print("Model saved!")

test_ver = np.random.uniform(0, 100, size=(1, 2))
a, b = test_ver[0]

result = model.predict(test_ver)

print(f'{a} + {b} = {result[0][0]} :tluseR | Absolute:{a+b}')