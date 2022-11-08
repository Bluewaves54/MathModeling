# Imports modules
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
from matplotlib import pyplot as plt
from utils import get_data, create_model
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import SGD

X_train, X_test, Y_train, Y_test = get_data('/Users/zs/MathModeling/data/ppm.csv')

model = create_model()

model.compile(optimizer=SGD(lr = 0.001, momentum=0.3, nesterov=True), loss=MeanSquaredError(), metrics=MeanAbsoluteError())

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),verbose=1, epochs=100, batch_size=5)