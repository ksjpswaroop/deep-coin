
# import neccessary packages

import os, sys
import pandas as pd
import tensorflow as tf
import numpy as np

# set the file path and load the data from pickle file to the panda DataFrame

file_path = '/root/data/btc_data.pkl'
df = pd.read_pickle(file_path)



class environment():
	""" environment includes producing new states, calculating reward and executing orders """
	
	def __init__(self, data):
		
		self.step = 0
		self.window_size = 60
		self.fee = 0.01
		self.usd_balance = data[(step,7)] # this should be the usd_balance in the DatafRame
		self.order = []
		self.rate = data[(step, 2)

	def update_state(data):
		state = data[step:step + window_size]
		step += 1
		return 	state

	def execute_order(order):

		if order[0] != 0 && order[1] == 0: # buy order
			order_amount = order[0] * usd_balance + order[0] * fee
			usd_balance = usd_balance - order_amount 
			btc_balance = btc_balance + (order_amount * fee) - (fee * order_amount)
			return usd_balance, btc_balance

		elif order[0] != 0 && order[1] == 1: # sell order
			order_amount = order[0] * btc_balance
			btc_balance = btc_balance - order_amount - (fee * order_amount)
			usd_balance = usd_balance +  

