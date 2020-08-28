import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam

#download data
(X_train,y_train),(X_test,y_test) = mnist.load_data()

X_train = X_train.reshape(-1,1,28,28)/255
X_test = X_test.reshape(-1,1,28,28)/255

y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

#build CNN
model = Sequential()
model.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding="same",
    data_format="channels_first",
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding="same",
    data_format="channels_first",
))

model.add(Convolution2D(filters=64, kernel_size=5, strides=(1,1), padding="same", data_format="channels_first", activation="relu"))
model.add(MaxPooling2D(2,2,"same",data_format="channels_first"))

model.add(Flatten())
model.add(Dense(1024,activation="relu"))

model.add(Dense(10,activation="softmax"))

adam = Adam(lr=1e-4)

model.summary()

model.compile(optimizer=adam,
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print("training--------------")

model.fit(X_train,y_train, epochs=5, batch_size=32)

print("\n testing---------------")

loss, accuracy = model.evaluate(X_test,y_test)

print("\n test loss", loss)
print("\n test accuracy", accuracy)
