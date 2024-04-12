import os
import tensorflow
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import pickle

X_val = pickle.load(open("X_val.pickle","rb"))
y_val = pickle.load(open("y_val.pickle","rb"))

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = tf.keras.utils.normalize(X, axis=1)
X_val = X_val/255.0


model = tf.keras.models.Sequential()

#model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape = X.shape[1:], activation = "relu"))
model.add(tf.keras.layers.Flatten(input_shape = (50,50)))
model.add(tf.keras.layers.Dense(128, activation = "relu"))
model.add(tf.keras.layers.Dense(128, activation = "relu"))
model.add(tf.keras.layers.Dense(1, activation = "softmax"))

model.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics = ["accuracy"])

model.fit(X, y, epochs=3)

model.save("shape.model")


model = tf.keras.models.load_model("shape.model")

loss, accuracy = model.evaluate(X_val, y_val)

image_number = 1
while os.path.isfile(f"Testing/{image_number}.png"):

    try:
        img = cv2.imread(f"Testing/{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = np.argmax(model.predict(img))
        print(f"The shape is: {prediction}")
        plt.imshow(img[0], cmap="gray")
        plt.show()

    except:
        print("error")

    finally:
        image_number += 1




