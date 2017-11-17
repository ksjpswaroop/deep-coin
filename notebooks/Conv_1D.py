# coding: utf-8
# Deep-Coin
# the trading agent
# Baseline architecture with 1D Convolutional network using keras API and Tensorflow backend



# import neccesary packages for this project
from __future__ import print_function, division
import gc
import sys
import os
import math
import datetime as dt
import numpy as np
import pandas as pd
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from keras.models import Sequential



# Data prepration
def load_data(file_path, drop_na=True, utc_time=True):
	"""this function will load the csv data file into a panda dataframe and converts the date-time to utc standard if True and returns the df"""
	df = pd.read_csv(file_path, parse_dates=True, keep_date_col=True, index_col=0, infer_datetime_format=True, error_bad_lines=False)
	if drop_na == True: 
		df = df.dropna()
	if utc_time == True:
		df.index = pd.to_datetime(df.index, utc=True, unit='s')
	return df

# reward function
def get_reward(a, b):
	"""it will calculate the diffrence between portfolio value at state a and b"""
	return (b-a) / a

# convolutional 1D model with 2 convolutional layers, 2 maxpooling, 1 flatten and 1 dense network
def build_convolution1D_model(window_size, filter_length, nb_input_series=1, nb_outputs=1, nb_filter=4):
    model = Sequential((
        Conv1D(filters=nb_filter, kernel_size=filter_length, activation='relu', input_shape=(window_size, nb_input_series)),
        MaxPooling1D(), # Downsample the output of convolution by 2X.
        Conv1D(filters=nb_filter, kernel_size=filter_length, activation='relu'),
        MaxPooling1D(),
	Conv1D(filters=nb_filter, kernel_size=filter_length, activation='relu'),
        MaxPooling1D(),
	Flatten(),
        Dense(nb_outputs, activation='linear'))) # For binary classification, change the activation to 'sigmoid'
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

# path to the csv file 
file_path = '~/data/bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv'

# data prepration
df = load_data(file_path)
df = df.drop(['Open', 'High', 'Low', 'Volume_(Currency)', 'Weighted_Price'], axis=1)
df['BTC_balance'] = 0
df['USD_balance'] = 1000
df['Portfolio_value'] = 0
df['total_rewards'] = 0
print (df.head())
print (df.tail())

# create the training, validation and test datasets
training_size = 1000000
validation_size = 500000
test_size = 150904
train_df = df[:training_size]
valid_df = df[training_size:training_size+validation_size]
test_df = df[training_size+validation_size:]
print(len(train_df), len(valid_df), len(test_df))

# model's hyperparameters
window_size = 60
nb_filter = 16
filter_length = 5
nb_input_series = 6
nb_outputs = 2

# build the model and print the summary of the model
model = build_convolution1D_model(window_size=window_size, filter_length=filter_length, nb_input_series=nb_input_series, nb_outputs=nb_outputs, nb_filter=nb_filter)
print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape, model.output_shape, nb_filter, filter_length))
model.summary()
