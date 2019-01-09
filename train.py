import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


animals = np.load("data-numpy/animals.npy")
labels = np.load("data-numpy/labels.npy")

# Shuffle the data to get a good mixture of the data set
s = np.arange(animals.shape[0])
np.random.shuffle(s)
animals = animals[s]
labels = labels[s]

# Number of animals type
numberOfAnimalsTypes = 2

# Data set size
dataSetSize = len(animals)

# Split data into test and train
(x_train, x_test) = animals[int(0.1 * dataSetSize):], animals[:int(0.1 * dataSetSize)]

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

trainLength = len(x_train)
testLength = len(x_test)

# Split labels into test and train
(y_train, y_test) = labels[int(0.1 * dataSetSize):], labels[:int(0.1 * dataSetSize)]

# One hot encoding
y_train = keras.utils.to_categorical(y_train, numberOfAnimalsTypes)
y_test = keras.utils.to_categorical(y_test, numberOfAnimalsTypes)

# Make model
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=2, padding="same", activation="relu", input_shape=(50, 50, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2, activation="softmax"))
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=50, epochs=30, verbose=1)

# Test the  model
score = model.evaluate(x_test, y_test, verbose=1)
print('\nTest  accuracy: ', score[1])

# Save the model
model_json = model.to_json()
with open("models/model.json", "w") as jsonFile:
    jsonFile.write(model_json)
model.save_weights("models/model.h5")
print("Saved model to disk")
