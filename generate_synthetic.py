import numpy as np
import cv2
import os
import uuid
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your collected digits
SOURCE_PATH = "my_digits"
TARGET_PATH = "synthetic_digits"

os.makedirs(TARGET_PATH, exist_ok=True)

# Strong augmentation settings
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    zoom_range=0.3,
    shear_range=20,
    brightness_range=[0.7, 1.3]
)

TARGET_PER_IMAGE = 1000  # generates 1000 synthetic per original

for digit in os.listdir(SOURCE_PATH):

    digit_path = os.path.join(SOURCE_PATH, digit)
    if not os.path.isdir(digit_path):
        continue

    save_path = os.path.join(TARGET_PATH, digit)
    os.makedirs(save_path, exist_ok=True)

    for img_name in os.listdir(digit_path):

        img_path = os.path.join(digit_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = img.reshape((1, 28, 28, 1))

        count = 0
        for batch in datagen.flow(img, batch_size=1):

            augmented = batch[0].reshape(28,28)

            filename = os.path.join(
                save_path,
                f"{uuid.uuid4().hex}.png"
            )

            cv2.imwrite(filename, (augmented * 255).astype("uint8"))

            count += 1
            if count >= TARGET_PER_IMAGE:
                break

print("Synthetic dataset generated!")
