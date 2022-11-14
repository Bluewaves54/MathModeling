# Imports modules
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
from matplotlib import pyplot as plt
from utils import format_ppm_x, create_model
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from sklearn.model_selection import train_test_split
import tensorflow as tf

NUM_PREV_VALS = 7

data = pd.read_csv(r'/Users/zs/MathModeling/data/filtered_co2.csv')
data = format_ppm_x(data, NUM_PREV_VALS)

X = data.iloc[:, data.columns != 0]
Y = data[0]

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.1)

model = create_model()

optim = tf.keras.optimizers.Adagrad(learning_rate=0.01,
    initial_accumulator_value=0.12)

model.compile(optim, loss = 'mae', metrics = 'mse')

history = model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test, y_test),
    batch_size=16,
    epochs=300)