import os.path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people


model_cnn_file = r'model_cnn.keras'

print("Загружаем CIFAR-10...")
(cifar_images, cifar_labels), (cifar_test, cifar_test_l) = tf.keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
animals = [2, 3, 4]
animal_mask = np.isin(cifar_labels.flatten(), animals)
animal_images = cifar_images[animal_mask]
animal_labels = np.ones((animal_images.shape[0], 1))
print(f"Найдено животных: {animal_images.shape[0]}")

print("Загружаем LFW...")
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
lfw_images = lfw_people.images
print(f"Размер LFW images: {lfw_images.shape}")

def resize_img(images):
    resized_img = []
    for img in images:
        image_resized = cv2.resize(img, (32, 32))
        image_rgb = np.stack([image_resized] * 3, axis=-1)
        resized_img.append(image_rgb)
    return np.array(resized_img)

lfw_images_resized = resize_img(lfw_images)
lfw_labels = np.zeros((lfw_images_resized.shape[0], 1))  

print(f"Размер animal_images: {animal_images.shape}")
print(f"Размер lfw_images_resized: {lfw_images_resized.shape}")

db_images = np.concatenate([animal_images, lfw_images_resized], axis=0)
db_labels = np.concatenate([animal_labels, lfw_labels], axis=0)

print(f"Общий размер датасета: {db_images.shape}")
indices = np.random.permutation(len(db_images))
db_images = db_images[indices]
db_labels = db_labels[indices]
db_images = db_images / 255.0

x_train, x_test, y_train, y_test = train_test_split(
    db_images, db_labels, test_size=0.2, random_state=42, stratify=db_labels)
print(f"Train: {x_train.shape}, Test: {x_test.shape}")

if os.path.exists(model_cnn_file):
    print("Loading model file...")
    model_cnn = tf.keras.models.load_model(model_cnn_file)
    print("Model is loaded")
else:
    model_cnn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),  
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model_cnn.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    model_cnn.summary()

    print("Начинаем обучение...")
    history = model_cnn.fit(
        x_train, y_train,
        epochs=10,  
        batch_size=32,
        validation_data=(x_test, y_test),
        verbose=1
    )
    model_cnn.save(model_cnn_file)

model_cnn.summary()
test_loss, test_accuracy, test_precision, test_recall = model_cnn.evaluate(x_test, y_test, verbose=0)
print(f"\n=== РЕЗУЛЬТАТЫ ===")
print(f"Точность: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")

def show_predictions(model, images, labels, num_samples=8):
    predictions = model.predict(images[:num_samples])
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i])
        pred_prob = predictions[i][0]
        true_label = labels[i][0]
        pred_class = "Животное" if pred_prob > 0.5 else "Человек"
        true_class = "Животное" if true_label == 1 else "Человек"
        color = 'green' if pred_class == true_class else 'red'
        confidence = max(pred_prob, 1 - pred_prob)
        plt.title(f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}',
                  color=color, fontsize=9)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

show_predictions(model_cnn, x_test, y_test)