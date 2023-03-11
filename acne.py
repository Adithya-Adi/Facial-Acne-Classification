import tkinter as tk
from tkinter import filedialog, LEFT, TOP, RIGHT, BOTTOM
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

def submit_callback():
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    image = image.resize((224,224))
    image = np.array(image)/255.0
    preview = ImageTk.PhotoImage(Image.open(file_path).resize((200, 200), Image.ANTIALIAS))
    image_label.config(image=preview)
    image_label.image = preview

def classify_callback():
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    label.config(text="Prediction: " + str(predictions), font=("Arial", 14), bg="#e6e6fa")

root = tk.Tk()
root.title("Machine Learning GUI")
root.geometry("800x600")
root.config(bg="#e6e6fa")

browse_button = tk.Button(root, text="Select Image", command=submit_callback, font=("Arial", 16), padx=20, pady=10, bg="#800080", fg="white")
browse_button.pack(side=LEFT, padx=10, pady=10)

classify_button = tk.Button(root, text="Classify", command=classify_callback, font=("Arial", 16), padx=20, pady=10, bg="#800080", fg="white")
classify_button.pack(side=BOTTOM, padx=10, pady=10)

label = tk.Label(root, text="Please select an image to make a prediction", font=("Arial", 14), bg="#e6e6fa")
label.pack(side=TOP, padx=10, pady=10)

image_label = tk.Label(root, width=600, height=600, bg="#e6e6fa")
image_label.pack(side=RIGHT, padx=10, pady=10)

model = tf.keras.models.load_model('model/model.h5')

root.mainloop()
