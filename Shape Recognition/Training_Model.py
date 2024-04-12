import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "Datasets"
CATEGORIES = ["Circles", "Triangles"]


training_data = []


def create_training_data():

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):

            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                training_data.append([img_array, class_num, path])

            except:
                pass

create_training_data()

random.shuffle(training_data)


pickle_out = open("Training.pickle", "wb")
pickle.dump(training_data, pickle_out)
pickle_out.close()


##X = []
##y = []
##
##for features, label in training_data:
##    X.append(features)
##    y.append(label)
##
##X = np.array(X).reshape(-1, 50,50, 1)
##y = np.array(y)
##
##pickle_out = open("X.pickle", "wb")
##pickle.dump(X, pickle_out)
##pickle_out.close()
##
##pickle_out = open("y.pickle", "wb")
##pickle.dump(y, pickle_out)
##pickle_out.close()




