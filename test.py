import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


model = tf.leras.models.load_model('hw.keras')

image_number = 1

while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)

        print(f"This digis is propably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    catch:
        print("ERROR!")
    finally:
        image_number += 1
