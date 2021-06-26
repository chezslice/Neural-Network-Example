import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt

# Importing Keras dataset.
data = keras.datasets.fashion_mnist

# Split data into the training and test data. To test accuracy.
(images_train, labels_train), (images_test, labels_test) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Greyscale each image to 255. This is done for reasons to scale down value of each image to make computations more easier.
images_train = images_train/255.0
images_test = images_test/255.0

# Creating the nueral network, using sequential object from keras starting with input and output layer, 
# a hidden layer is then added with a value of 128 neurons.
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax")
	])

# Compile and train the model. We train the data 10 times.
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=
["accuracy"])

model.fit(images_train, labels_train, epochs=10)

# Testing the model for its accuracy.
loss_test, acc_test = model.evaluate(images_test, labels_test)
print('\nTest accuracy:', acc_test)

# Using the model, we predict the testing images. Multiple predictions are made from the data inside of a list.
predictions = model.predict(images_test)

plt.figure(figsize=(5,5))
for i in range(5):
    plt.grid(False)
    plt.imshow(images_test[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels_test[i]])
    plt.title(class_names[np.argmax(predictions[i])])
    plt.show()
