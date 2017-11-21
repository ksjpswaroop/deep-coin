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

# loss function returns  value between -1 and 1 for gain or loss.
def get_loss (x, y): # y=current value of portfolio, x=previous state's value of portfolio
	"""it will calculate the diffrence between portfolio value at state a and b"""
	return -1 * ((y - x) / (x + y))

# environment gets the output of policy network and returns reward and the next observation
def env (sate, actions, fee=transaction_fee): 	# if actions is a 2x1 matrix, lets say index 0 is buy signal and index 1 is sell signal 
	
	action = actions[0] + actions[1]	# with probability of their value. 
	
	if action * state[3] > transaction_fee * state[0]: # current_state[0] is the price st the time
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
		
		  
# convolutional 1D model with 2 convolutional layers, 2 maxpooling, 1 flatten and 1 dense network
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
	
