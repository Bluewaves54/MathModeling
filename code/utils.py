import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense, Dropout
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

def create_model():
    model = Sequential()
    model.add(LSTM(64, input_shape = (62,2), dropout = 0.2, return_sequences=True))
    model.add(LSTM(10, dropout = 0.2, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation = "relu"))
    return model

