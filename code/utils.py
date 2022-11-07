import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense, InputLayer
from keras import Sequential

# Loads dataset and seperates into X and Y
scaler = MinMaxScaler()
def get_data(data):
    dataset = pd.read_csv(data, on_bad_lines='skip')
    X = np.array(dataset.Year[:-1]).reshape(-1,1)
    X = scaler.fit_transform(X)
    Y = dataset['PPM'][1:]
    X_new = np.zeros((len(X),len(X),2))
    for i in range(len(X)):
        for j in range(i):
            X_new[i,(len(X)-j-1),0] = X[i-j]
            X_new[i,(len(X)-j-1),1] = Y[i-j]
    X = X_new
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    return X_train, X_test, Y_train, Y_test


def create_model(size):
    model = Sequential()
    model.add(InputLayer(size[1:]))
    model.add(LSTM(128, activation = 'relu', dropout = 0.2,))
    model.add(LSTM(64, activation = 'relu', dropout = 0.2,))
    model.add(LSTM(32, activation = 'relu', dropout = 0.2,))
    model.add(Dense(1, activation = "relu"))
    return model
