import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense, InputLayer, BatchNormalization
from keras import Sequential
from string import ascii_lowercase

# Loads dataset and seperates into X and Y
scaler = MinMaxScaler()
# def get_data(data):
#     dataset = pd.read_csv(data, on_bad_lines='skip')
#     X = np.array(dataset.Year[:-1]).reshape(-1,1)
#     X = scaler.fit_transform(X)
#     Y = dataset['PPM'][1:]
#     X_new = np.zeros((len(X),len(X),2))
#     for i in range(len(X)):
#         for j in range(i):
#             X_new[i,(len(X)-j-1),0] = X[i-j]
#             X_new[i,(len(X)-j-1),1] = Y[i-j]
#     X = X_new
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#     return X_train, X_test, Y_train, Y_test

NUM_PREV_VALS = 7

def create_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(NUM_PREV_VALS,1)))
    # model.add(LSTM(128, activation='relu', dropout=0.05, return_sequences=True))
    model.add(LSTM(16, activation='relu', dropout=0.1, return_sequences=True))
    model.add(Dense(1, activation='relu'))
    return model

def format_ppm_x(data:pd.DataFrame, num_previous_vals: int):
    iterator, new_data = list(data.iterrows())[num_previous_vals:], {i: [] for i in range(num_previous_vals+1)[::-1]}
    print('checkpoint 1')
    for index, val in iterator:
        for i in range(num_previous_vals+1)[::-1]:
            new_data[i].append(data.iloc[index-i][1])
   
    return pd.DataFrame(new_data)


# def decompose(data):
#     eemd_decomp = {ascii_lowercase[i]: val for i, val in enumerate(EEMD(data))}
#     vmd_decomp = {f'a{i}': val for i, val in VMD(eemd_decomp[0])}
#     full_decomp = eemd_decomp | vmd_decomp