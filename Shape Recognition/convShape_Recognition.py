import tensorflow
import tensorflow as tf
import pickle
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense

training_model = pickle.load(open("training.pickle", "rb"))

X = []
y = []
name = []

for features, label, path in training_model:
    X.append(features)
    y.append(label)
    name.append(path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train = np.array(X_train).reshape(-1, 50,50, 1)
X_test = np.array(X_test).reshape(-1, 50,50, 1)

y_train = np.array(y_train)
y_test = np.array(y_test)


X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)


###-----Model
##
##model = Sequential()
##model.add(Conv2D(32, (3,3), input_shape= (50,50, 1)))
##model.add(Activation("relu"))
##model.add(MaxPooling2D(pool_size = (2,2)))
##
##model.add(Conv2D(32, (3,3), kernel_initializer= "he_uniform"))
##model.add(Activation("relu"))
##model.add(MaxPooling2D(pool_size = (2,2)))
##
##model.add(Conv2D(64, (3,3), kernel_initializer= "he_uniform"))
##model.add(Activation("relu"))
##model.add(MaxPooling2D(pool_size = (2,2)))
##
##model.add(Flatten())
##model.add(Dense(64))
##model.add(Activation("relu"))
##model.add(Dropout(0.5))
##
##model.add(Dense(1))
##model.add(Activation("sigmoid"))
##
##
##model.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])
##
##model.fit(X_train, y_train, verbose = 1, epochs = 5, validation_data= (X_test, y_test))
##
##model.save("convShape.model")

model = tf.keras.models.load_model("convShape.model")


for n in range(20):
    img = X_test[n]

    input_img = np.expand_dims(img, axis=0)
    print("prediction", model.predict(input_img))
    print("actual", y_test[n])

    plt.imshow(img, cmap = "gray")
    plt.show()

