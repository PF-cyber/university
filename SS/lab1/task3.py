import tensorflow as tf
import numpy as np
import os

model_path = 'model_file_three_sum.keras'

if os.path.exists(model_path):
    print("Loading model file...")
    model = tf.keras.models.load_model(model_path)
    print("Model is loaded!")
else:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.summary()

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )

    x = np.random.uniform(0, 100, size=(300, 3))
    y = np.sum(x, axis=1)

    print("Learning...")
    history = model.fit(x, y, epochs=10000, verbose=False)

    print(f"Save model: {model_path}")
    model.save(model_path)
    print("Model saved!")

test_var = np.random.uniform(0, 300, size=(1, 3))
a, b, c = test_var[0]

result = model.predict(test_var, verbose=False)

print(f'{a} + {b} + {c} = {result[0][0]} :tluseR | Absolute:{sum((a,b,c))}')