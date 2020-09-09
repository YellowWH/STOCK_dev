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
from keras.models import Sequential, load_model
from keras import backend
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, LSTM, Dropout
from keras.optimizers import Adam, RMSprop
from keras import initializers
import random
import matplotlib.pyplot as plt
import time

X = np.load("X.npy")
Y = np.load("Y.npy")
tianshu = 30

# 训练回数
epoch_num = 500
dropout = 0
png_name = "pyplot//epoch_" + str(epoch_num) + "drop_" + str(dropout) + "0901分類7.png"
svg_name = "pyplot//epoch_" + str(epoch_num) + "drop_" + str(dropout) + "0901分類7.svg"


# 分组
asd = max(Y)
print(Y[Y <= -0.0], Y[Y >= 0.2])
Y = np.clip(Y, 0.0, 0.2)
print(Y[Y <= -0.0], Y[Y >= 0.2])
Y_category = []
for i in range(0, Y.size):
    temp = [0] * 21
    index = int(Y[i] * 100)
    temp[index] = 1
    Y_category.append(temp)
Y_category_new = np.array(Y_category)

Y = Y * 100

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
randomTestList = random.sample(range(0, X.shape[0]), 1000)
X_test = X[randomTestList]
X_test1 = X_test.reshape(1000, tianshu * 9)
Y_test = Y_category_new[randomTestList]
X_train = np.delete(X, randomTestList, 0)
Y_train = np.delete(Y_category_new, randomTestList, 0)

model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(tianshu, 9)))
model.add(Dropout(rate=dropout))
model.add(Dense(1024, activation="relu"))
model.add(Dense(21, activation="softmax"))
adam = Adam()
rmsprop = RMSprop()
model.compile(optimizer=rmsprop,
              loss="mse",
              metrics=["acc"])
model.summary()

print("training--------------")
# model.fit(X_train, Y_train, epochs=5, batch_size=128)

history = model.fit(X_train, Y_train, epochs=epoch_num, batch_size=128)

# 画曲线
acc = history.history['acc']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)

# plt.title('Accuracy and Loss')
# l1 = plt.plot(epochs, acc, 'red', label='Training acc')
# l2 = plt.plot(epochs, loss, 'blue', label='Validation loss')
# plt.legend()
# plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
l1 = plt.plot(epochs, acc, 'red', label='Training acc', linewidth=2)
ax1.set_ylabel('Test accuracy', fontsize=15)
ax2 = ax1.twinx()
l2 = ax2.plot(epochs, loss, 'blue', label='Validation loss', linewidth=2)
ax2.set_xlim(left=0, right=epoch_num)
ax2.set_ylim(0.0000, 3.0000)
ax2.set_ylabel('Loss', fontsize=15)
lns = l1 + l2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='center right')
plt.show()
fig.savefig(png_name, dpi=600)
fig.savefig(svg_name, dpi=600, format='svg')

print("\n testing---------------")
loss, accuracy = model.evaluate(X_test, Y_test)
print("\n test loss", loss)
print("\n test accuracy", accuracy)


today = time.strftime("%Y-%m-%d",time.localtime(time.time()))
model.save('model_weight//'+today+'model_weight.h5')

