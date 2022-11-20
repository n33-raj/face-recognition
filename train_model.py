### Importing Libraries

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.layers import Dense,Input,Dropout,GlobalAveragePooling2D,Flatten,Conv2D,BatchNormalization,Activation,MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD,RMSprop



EPOCHS  = 50

## Preprocessing the training data
train_datagen = ImageDataGenerator(rescale = 1./255,              ## data agumentation
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range=20,
                                   horizontal_flip = True,
                                   fill_mode = "nearest")

training_set = train_datagen.flow_from_directory(r'dataset\train',
                                                 target_size = (48,48),
                                                 color_mode = "grayscale",          ## changing images to grayscale
                                                 batch_size = 32,
                                                 class_mode = "categorical",
                                                 shuffle = True)

## Preprocessing the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(r'dataset\test',
                                            target_size = (48,48),
                                            color_mode = "grayscale",
                                            batch_size = 32,
                                            class_mode = "categorical",
                                            shuffle = False)

### Building CNN
cnn = models.Sequential()                  ## initialising cnn

## 1st layer
cnn.add(layers.Conv2D(filters=32,                
                    kernel_size=3,        ## means 3x3
                    padding = 'same',
                    activation='relu',
                    input_shape=[48,48,1]))        ### 1: for gray color 

cnn.add(layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(BatchNormalization())


## 2nd layer
cnn.add(layers.Conv2D(filters=64,
                    kernel_size=5,
                    padding = 'same',
                    activation='relu'))

cnn.add(layers.MaxPool2D(pool_size=2,strides=2))

## 3rd layer
cnn.add(layers.Conv2D(filters=128,
                    kernel_size=3,
                    padding = 'same',
                    activation='relu'))

cnn.add(layers.MaxPool2D(pool_size=2,strides=2))

## 4th layer
cnn.add(layers.Conv2D(filters=256,
                    kernel_size=3,
                    padding = 'same',
                    activation='relu'))

cnn.add(layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(layers.Dropout(rate=0.2))



## Flattening
cnn.add(layers.Flatten())


## 1st Full Connection layer
cnn.add(layers.Dense(units=256, activation='relu'))
cnn.add(BatchNormalization())



## 2nd Full Connection layer
cnn.add(layers.Dense(units=512, activation='relu'))

## 3rd Full Connection layer
cnn.add(layers.Dense(units=256, activation='relu'))
cnn.add(layers.Dropout(rate=0.2))



## Output Layer
cnn.add(layers.Dense(units=4, activation='softmax'))     ## as we have 4 classes in train data

### Training The CNN
opt = Adam(lr = 0.0001)
cnn.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])     ## compiling the CNN


### Model Summary
cnn.summary()

### Training the CNN on the Training set and evaluating it on the validation set
H = cnn.fit(training_set, validation_data = test_set, batch_size = None, epochs = EPOCHS)

## Saving Model
cnn.save('Face-Recognition-Model.h5')


## plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0,N), H.history["loss"], label="Training Loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="Validation Loss")
plt.plot(np.arange(0,N), H.history["accuracy"], label="Training Accuracy")
plt.plot(np.arange(0,N), H.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

## save plot to disk
plt.savefig('plot.png')