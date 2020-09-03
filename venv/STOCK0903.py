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
from keras.optimizers import Adam
from keras import initializers
import random
import matplotlib.pyplot as plt
import time


X_real = np.load("X_real.npy")
X_code = np.load("X_code.npy")
today = time.strftime("%Y-%m-%d",time.localtime(time.time()))
model = load_model('model_weight//'+today+'model_weight.h5')

print(model.predict(X_real))
print(model.predict_classes(X_real))



print("q")
