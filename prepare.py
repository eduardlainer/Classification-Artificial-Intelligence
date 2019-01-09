import cv2
import os

from PIL import Image
import numpy as np

"""

Read images
Data contains re-sized images for cats and dogs.
Labels 0 for cats and 1 for dogs

"""
print("Preparing data set . . .")

data = []
labels = []

# Cats
catsFolder = os.listdir("images/cats/")
for catImage in catsFolder:
    print("cats" + catImage)
    image = cv2.imread("images/cats/" + catImage)
    if image is None:
        break
    imageFromArray = Image.fromarray(image, 'RGB')
    resizedImage = imageFromArray.resize((50, 50))
    data.append(np.array(resizedImage))
    labels.append(0)

# Dog
dogsFolder = os.listdir("images/dogs/")
for dogImage in dogsFolder:
    print("dogs" + dogImage)
    image = cv2.imread("images/dogs/" + dogImage)
    if image is None:
        break
    imageFromArray = Image.fromarray(image, 'RGB')
    resizedImage = imageFromArray.resize((50, 50))
    data.append(np.array(resizedImage))
    labels.append(1)


animals = np.array(data)
labels = np.array(labels)

np.save("data-numpy/animals", animals)
np.save("data-numpy/labels", labels)

print("Data set prepared!")
