import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import cv2

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from tensorflow.python.keras.testing_utils import layer_test

(cifar_images, cifar_labels), (cifar_test, cifar_test_l) = tf.keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
animals = [2, 3, 4]

animal_mask = np.isin(cifar_labels.flatten(), animals)
animal_images = cifar_images[animal_mask]
animal_labels = np.ones((animal_images.shape[0], 1))

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
lfw_images = lfw_people.images


def resize_img(img):
    resized_img = []
    for i in img:
        image = cv2.resize(img, (32, 32))
        image = np.stack([image] * 3, axis=1)
        resized_img.append(image)
    return np.array(resized_img)


lfw_images = np.squeeze(resize_img(lfw_images).shape[0], axis=1)
lfw_labels = np.ones((lfw_images.shape[0], 1))

db_images = np.concatenate([animal_images, lfw_images], axis=0)
db_labels = np.concatenate([animal_labels, lfw_labels], axis=0)

indices = np.random.permutation(len(db_images))
db_images = db_images[indices]
db_images = db_images / 255.0

x_train, x_test, y_train, y_test = train_test_split(db_images, db_labels, test_size=0.2, random_state=42,
                                                    stratify=db_labels)

model_cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model_cnn.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'precision', 'recall'])

model_cnn.summary()

