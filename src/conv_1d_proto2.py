
# coding: utf-8

# In[13]:


import os, sys, gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from keras.models import Sequential

#sys.path.append('/root/src')
#from model_conv_1d import load_data, build_convolution1D_model, get_loss
file_path = '~/data/bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv'


# model's hyperparameters
window_size = 60
nb_filter = 10
filter_length = 10
nb_input_series = 5
nb_outputs = 2
epochs = 1
batch_size = 1
initial_bal = 1000
window_size = 60
num_steps = 1000000


def load_data(file_path, drop_na=True, utc_time=False):
    """this function will load the csv data file into a panda dataframe and converts the date-time to utc standard if True and returns the df"""
    df = pd.read_csv(file_path, parse_dates=True, keep_date_col=True, index_col=0, infer_datetime_format=True, error_bad_lines=False)
    if drop_na == True: 
        df = df.dropna()
    if utc_time == True:
        df.index = pd.to_datetime(df.index, utc=True, unit='s')
    return df


# data prepration
df = load_data(file_path)
df = df.drop(['Open', 'High', 'Low', 'Volume_(Currency)', 'Weighted_Price'], axis=1)
df['btc_bal'] = 0
df['usd_bal'] = initial_bal
df['portfolio_bal'] = 0
df['total_rewards'] = 0
print (df.shape)
df.head()



def get_state(data_frame, step, window_size=60):
    return data_frame[step:step + window_size]


def loss(previous_state_value, current_state_value):
    loss = -1 * (current_state_value - previous_state_value) / (current_state_value + previous_state_value)
    return loss


def env (sate, actions, fee): 	# if actions is a 2x1 matrix, lets say index 0 is buy signal and index 1 is sell signal 
    action = actions[0] + actions[1]	# with probability of their value.  
    if action * state[3] > fee * state[0]: # current_state[0] is the price st the time
        state[2] = state[2] + (action * state[3] / state[0]  - fee) # let's say idx 2 of current_state is btc_balance,
        state[3] = state[3] - action * state[3] - fee * state[0]
        state[4] = state[3] + state[2] * state[0]
        return state
    elif (action * state[2] * -1) > fee:
        state[2] = state[2] - (action * state[2])
        state[3] = state[3] + (action * state[2] - fee * state[0])
        state[4] = state[3] + state[2] * state[0]
        return state
    else:
        return state

    
def build_convolution1D_model(window_size, filter_length, nb_input_series=1, nb_outputs=1, nb_filter=4):
    model = Sequential((
        Conv1D(filters=nb_filter, kernel_size=filter_length, activation='relu', input_shape=(window_size, nb_input_series)),
        MaxPooling1D(), # Downsample the output of convolution by 2X.
        Conv1D(filters=nb_filter, kernel_size=filter_length, activation='relu'),
        MaxPooling1D(),
        Flatten(),
        Dense(nb_outputs, activation='linear'), # For binary classification, change the activation to 'sigmoid'
        ))
    model.compile(loss=loss, optimizer='adam', metrics=[loss]) # this part needs correction, instead of loss function the value of loss function should be passed, or maybe not, i need to double check this.
    return model


# build the model and print the summary of the model
model = build_convolution1D_model(window_size=window_size, filter_length=filter_length, nb_input_series=nb_input_series, nb_outputs=nb_outputs, nb_filter=nb_filter)
print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape, model.output_shape, nb_filter, filter_length))
model.summary()
