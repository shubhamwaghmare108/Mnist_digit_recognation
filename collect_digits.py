import tkinter as tk
from PIL import ImageGrab
import numpy as np
import cv2
import os
import uuid

# Create main window
root = tk.Tk()
root.title("Digit Dataset Collector")

canvas = tk.Canvas(root, width=300, height=300, bg='black')
canvas.pack()

# Draw with mouse
def draw(event):
    x, y = event.x, event.y
    canvas.create_line(
        x, y, x+1, y+1,
        fill='white',
        width=8,
        capstyle=tk.ROUND,
        smooth=True
    )

canvas.bind("<B1-Motion>", draw)

# Preprocess to MNIST format
def preprocess_image(img):
    img = np.array(img)
    img = 255 - img
    img = cv2.resize(img, (28, 28))
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = img / 255.0
    return img

# Save function
def save_digit():
    label = label_entry.get()

    if label not in [str(i) for i in range(10)]:
        result_label.config(text="Enter digit 0-9")
        return

    # Capture canvas
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    img = ImageGrab.grab().crop((x, y, x1, y1))
    img = img.convert('L')

    processed = preprocess_image(img)

    # Create folder if not exists
    folder_path = f"my_digits/{label}"
    os.makedirs(folder_path, exist_ok=True)

    # Save image
    filename = f"{folder_path}/{uuid.uuid4().hex}.png"
    cv2.imwrite(filename, (processed * 255).astype("uint8"))

    result_label.config(text=f"Saved digit {label}")
    canvas.delete("all")

# Clear canvas
def clear_canvas():
    canvas.delete("all")
    result_label.config(text="")

# UI elements
label_entry = tk.Entry(root)
label_entry.pack()

btn_save = tk.Button(root, text="Save Digit", command=save_digit)
btn_save.pack()

btn_clear = tk.Button(root, text="Clear", command=clear_canvas)
btn_clear.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
