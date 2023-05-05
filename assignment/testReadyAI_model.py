import os
import tkinter as tk
import tensorflow as tf
from tensorflow import keras
from tkinter import filedialog
import numpy as np
from PIL import Image

# Create a tkinter window and hide it
root = tk.Tk()
root.withdraw()

# Ask the user to select an image file using a file dialog box
file_path = filedialog.askopenfilename()

# Load the selected image
img = Image.open(file_path)

# Preprocess the image
img = img.convert('L')  # convert to grayscale
img = img.resize((28, 28))  # resize to (28, 28)
img_arr = np.array(img)  # convert to numpy array
img_arr = img_arr / 255.0  # normalize pixel values
img_arr = img_arr.reshape((1, 28, 28))  # reshape to (1, 28, 28)

# Load the model
model_path = 'assignment/readyModel/model.h5'
model = tf.keras.models.load_model(model_path)

label_map = {
    0: "tshirt/top",
    1: "trouser",
    2: "pullover",
    3: "dress",
    4: "coat",
    5: "sandal",
    6: "shirt",
    7: "sneaker",
    8: "bag",
    9: "ankle boot"
}

# Get the predicted label number
pred = model.predict(img_arr)
result = np.argmax(pred)

# Get the name of the predicted label
label_name = label_map[result]

print(f"The predicted item is: {label_name}, ({result} label)")