import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "Validating"
CATEGORIES = ["Circles", "Triangles"]

validating_data = []

def create_validating_data():

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):

            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                validating_data.append([img_array, class_num])

            except:
                pass

create_validating_data()

random.shuffle(validating_data)

X_val = []
y_val = []

for features, label in validating_data:
    X_val.append(features)
    y_val.append(label)

X_val = np.array(X_val).reshape(-1, 50,50, 1)
y_val = np.array(y_val)

pickle_out = open("X_val.pickle", "wb")
pickle.dump(X_val, pickle_out)
pickle_out.close()

pickle_out = open("y_val.pickle", "wb")
pickle.dump(y_val, pickle_out)
pickle_out.close()

