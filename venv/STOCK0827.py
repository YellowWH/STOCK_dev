import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

np.random.seed(1337)
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# 获取雅虎股票接口
import datetime
# 国内股票包
import tushare as ts
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, LSTM
from keras.optimizers import Adam
from keras import initializers
import random

X = np.load("X.npy")
Y = np.load("Y.npy")

# 还没做标准化！！！！！！！！！！

# 分组
asd = max(Y)
print(Y[Y <= -0.0], Y[Y >= 0.2])
Y = np.clip(Y, 0, 0.2)
print(Y[Y <= -0.0], Y[Y >= 0.2])
Y_category = []
for i in range(0, Y.size):
    temp = [0] * 21
    index = int(Y[i] * 100)
    temp[index] = 1
    Y_category.append(temp)
Y_category_new=np.array(Y_category)

# 洗牌
np.random.seed(10)
randomList = np.arange(X.shape[0])
np.random.shuffle(randomList)
X_length, X_width, X_depth = X.shape[0], X.shape[1], X.shape[2]
X_tem = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
X_tem = X_tem[randomList]
X = X_tem.reshape((X_length, X_width, X_depth))
Y_category_new = Y_category_new[randomList, :]

# 分开成训练组和test组
randomTestList = random.sample(range(0, X.shape[0]), 200)
X_test = X[randomTestList]
X_test1 = X_test.reshape(200, 30 * 9)
Y_test = Y_category_new[randomTestList]
X_train = np.delete(X, randomTestList, 0)
Y_train = np.delete(Y_category_new, randomTestList, 0)

model = Sequential()
model.add(LSTM(40, activation='relu', input_shape=(30, 9)))
model.add(Dense(21, activation="softmax"))
adam = Adam(lr=0.01)
model.compile(optimizer=adam,
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

print("training--------------")
model.fit(X_train, Y_train, epochs=1000, batch_size=128)

print("\n testing---------------")
loss, accuracy = model.evaluate(X_test, Y_test)
print("\n test loss", loss)
print("\n test accuracy", accuracy)
