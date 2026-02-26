import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("mnist_model_combined.h5")

def predict(data):
    try:
        if data is None:
            return "Draw a digit"

        # Handle Gradio Sketchpad output
        if isinstance(data, dict):
            image = data.get("composite", None)
            if image is None:
                return "No image data"
        else:
            image = data

        image = np.array(image)

        # Convert to grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # -------- MNIST STYLE PREPROCESSING --------

        # Invert (MNIST = white digit on black)
        gray = 255 - gray

        # Threshold to remove noise
        _, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Find digit bounding box
        coords = np.column_stack(np.where(gray > 0))
        if coords.size == 0:
            return "No digit detected"

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        digit = gray[y_min:y_max, x_min:x_max]

        h, w = digit.shape

        # Resize keeping aspect ratio to 20x20
        if h > w:
            new_h = 20
            new_w = int(w * (20 / h))
        else:
            new_w = 20
            new_h = int(h * (20 / w))

        digit = cv2.resize(digit, (new_w, new_h))

        # Create 28x28 blank canvas
        canvas_28 = np.zeros((28, 28))

        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2

        canvas_28[y_offset:y_offset+new_h,
                  x_offset:x_offset+new_w] = digit

        # Normalize
        canvas_28 = canvas_28.astype("float32") / 255.0

        input_image = canvas_28.reshape(1, 28, 28, 1)

        # Optional debug view
        plt.imshow(canvas_28, cmap='gray')
        plt.title("Processed Input")
        plt.show()

        # Prediction
        prediction = model.predict(input_image, verbose=0)
        digit_pred = np.argmax(prediction)
        confidence = np.max(prediction)

        return f"Prediction: {digit_pred} ({confidence:.2f})"

    except Exception as e:
        return f"Error: {e}"


interface = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(),
    outputs="text"
)

interface.launch()
