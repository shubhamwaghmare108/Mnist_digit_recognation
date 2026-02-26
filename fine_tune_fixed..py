import tensorflow as tf
import numpy as np
import os
import cv2

# ----------------------------
# 1️⃣ Load Your Custom Dataset
# ----------------------------

custom_images = []
custom_labels = []

dataset_path = "my_digits"

for label in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, label)

    if not os.path.isdir(folder):
        continue

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = img.astype("float32") / 255.0
        img = img.reshape(28, 28, 1)

        custom_images.append(img)
        custom_labels.append(int(label))

custom_images = np.array(custom_images)
custom_labels = np.array(custom_labels)

print("Custom samples loaded:", len(custom_images))

if len(custom_images) == 0:
    print("No images found. Check folder structure.")
    exit()

# ----------------------------
# 2️⃣ Load MNIST Dataset
# ----------------------------

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)

# ----------------------------
# 3️⃣ Combine MNIST + Custom
# ----------------------------

x_combined = np.concatenate((x_train, custom_images), axis=0)
y_combined = np.concatenate((y_train, custom_labels), axis=0)

# ----------------------------
# 4️⃣ Load Pretrained Model
# ----------------------------

model = tf.keras.models.load_model("mnist_model.h5")

# Freeze convolution layers only
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        layer.trainable = False

# Compile with very small learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------
# 5️⃣ Fine-Tune
# ----------------------------

model.fit(x_combined, y_combined, epochs=2, batch_size=64)

# ----------------------------
# 6️⃣ Save
# ----------------------------

model.save("mnist_model_finetuned.h5")

print("Fine-tuned model saved successfully!")
