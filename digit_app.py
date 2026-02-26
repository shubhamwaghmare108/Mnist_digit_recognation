import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("mnist_model_combined.h5")

def preprocess_digit(digit_img):
    digit_img = 255 - digit_img
    _, digit_img = cv2.threshold(digit_img, 50, 255, cv2.THRESH_BINARY)

    coords = np.column_stack(np.where(digit_img > 0))
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    digit = digit_img[y_min:y_max, x_min:x_max]

    h, w = digit.shape
    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))

    digit = cv2.resize(digit, (new_w, new_h))

    canvas_28 = np.zeros((28, 28))
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas_28[y_offset:y_offset+new_h,
              x_offset:x_offset+new_w] = digit

    canvas_28 = canvas_28.astype("float32") / 255.0

    return canvas_28.reshape(1, 28, 28, 1)

def predict(data):
    if data is None:
        return "Draw digits", None

    # Handle Gradio 4.x dict format
    if isinstance(data, dict):
        image = data.get("composite", None)
        if image is None:
            return "No image data", None
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

    # Threshold
   # Invert first (digits become white)
    gray_inv = 255 - gray

# Threshold
    _, thresh = cv2.threshold(gray_inv, 50, 255, cv2.THRESH_BINARY)

# Erode slightly to separate touching digits
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return "Draw digits", image

    # Sort left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    result = ""
    debug_image = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding box
        cv2.rectangle(debug_image, (x,y), (x+w, y+h), (0,255,0), 2)

        digit_region = gray[y:y+h, x:x+w]
        processed = preprocess_digit(digit_region)

        if processed is None:
            continue

        prediction = model.predict(processed, verbose=0)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        result += str(digit)

    return f"Prediction: {result}", debug_image


interface = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(
        height=600,   # 🔥 Bigger writing area
        width=400     # 🔥 Wider for multiple digits
    ),
    outputs=["text", "image"],
    title="Multi-Digit Recognition (MNIST Model)"
)

interface.launch()

interface.launch()
