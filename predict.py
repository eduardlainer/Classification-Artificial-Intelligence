import numpy as np
from PIL import Image

from keras.engine.saving import model_from_json
import cv2

from train import model

json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("models/model.h5")
print("Loaded model from disk")


def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((50, 50))
    return np.array(image)


def get_animal_name(label):
    if label == 0:
        return "cat"
    if label == 1:
        return "dog"


def predict_animal(file):
    print("Predicting .................................")
    ar = convert_to_array(file)
    ar = ar / 255
    a = [ar]
    a = np.array(a)
    score = model.predict(a, verbose=1)
    label_index = np.argmax(score)
    acc = np.max(score)
    animal = get_animal_name(label_index)

    print("The predicted animal is a " + animal + " with accuracy =    " + str(acc))


if __name__ == '__main__':
    images = ["dog.jpg",  "cat.jpeg", "dog2.jpg", "cat2.jpg", "cat3.jpg", "dog3.jpg"]
    for image in images:
        predict_animal("images/to-predict/"+image)
