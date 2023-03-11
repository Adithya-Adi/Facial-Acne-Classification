import tkinter as tk
from tkinter import filedialog, PhotoImage
import tensorflow as tf
import numpy as np
import cv2
import keras
from keras_preprocessing.image import ImageDataGenerator, img_to_array, load_img

class ImageClassificationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Facial Acne Classifier")

        # Add a background image
        self.background_image = PhotoImage(file="logo.png")
        self.background_label = tk.Label(self.master, image=self.background_image)
        self.background_label.pack()

        # Add a frame for the widgets
        self.frame = tk.Frame(self.master, bg="white")
        self.frame.pack(pady=50)
        self.model = tf.keras.models.load_model('model/model.h5')
        # Add a button to select an image
        self.select_image_button = tk.Button(self.frame, text="Select Image", command=self.select_image, bg="#009ACD", fg="white", font=("Helvetica", 16), padx=20, pady=10)
        self.select_image_button.pack(pady=20)

        # Add a label to display the selected image
        self.image_label = tk.Label(self.frame, text="", font=("Helvetica", 14))
        self.image_label.pack()

        # Add a button to classify the image
        self.classify_button = tk.Button(self.frame, text="Classify", command=self.classify, bg="#009ACD", fg="white", font=("Helvetica", 16), padx=20, pady=10)
        self.classify_button.pack(pady=20)

        # Add a label to display the classification result
        self.result_label = tk.Label(self.frame, text="", font=("Helvetica", 14))
        self.result_label.pack()

    def select_image(self):
        file_path = filedialog.askopenfilename()
        self.image_label.config(text=file_path)

    def classify(self):
        file_path = self.image_label["text"]
        if file_path:
            classifier_model = 'model/model.h5'
            model=keras.models.load_model(classifier_model)
            from keras import applications
            vgg16 = applications.VGG16(include_top=False, weights='imagenet')
            test_image = load_img(file_path, target_size=(224,224))
            test_image = np.array(test_image)
            test_image = test_image / 255.0
            test_image = np.expand_dims(test_image, axis=0)
            class_names = {0: 'Stage1:Comedones', 1:"Stage2:Papules",2:"Stage3:Pustules",3:"Stage4:Cysts"}
            bt_prediction = vgg16.predict(test_image)
            preds = model.predict(bt_prediction)
            labels = ['Stage1:Comedones','Stage2:Papules','Stage3:Pustules','Stage4:Cysts']
            for idx, animal, x in zip(range(0,6), labels , preds[0]):
                print("ID: {}, Label: {} {}%".format(idx, animal, round(x*100,2)))
            predictions=preds
            print(predictions)
            a=np.array(max(predictions))
            print(class_names[np.argmax(a)])
            result = f"Predicted as {class_names[np.argmax(a)]}"

            self.result_label.config(text=result)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassificationApp(root)
    root.mainloop()
