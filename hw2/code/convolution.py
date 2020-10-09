from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):
	"""
	Performs 2D convolution given 4D inputs and filter Tensors.
	:param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
	:param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
	:param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
	:param padding: either "SAME" or "VALID", capitalization matters
	:return: outputs, NumPy array or Tensor with shape [num_examples, output_height, output_width, output_channels]
	"""
	num_examples, in_height, in_width, input_in_channels = inputs.shape
	# num_examples = None
	# in_height = None
	# in_width = None
	# input_in_channels = None

	filter_height, filter_width, filter_in_channels, filter_out_channels = filters.shape
	# filter_height = None
	# filter_width = None
	# filter_in_channels = None
	# filter_out_channels = None

	num_examples_stride = strides[0]
	strideY = strides[1]
	strideX = strides[2]
	channels_stride = strides[3]

	assert input_in_channels == filter_in_channels, "ERROR: number of input in channels are not the same as the filters in channels"
	
	padY = 0
	padX = 0

	# Cleaning padding input
	if padding == "SAME":
		padY = (filter_height - 1) // 2
		padX = (filter_width - 1) // 2
		inputs = np.pad(inputs, ((0,0), (padY, padY), (padX, padX), (0,0)), mode='constant', constant_values=0.0)
		in_height += 2*padY
		in_width += 2*padX
	
	output = np.zeros((num_examples, (in_height - filter_height) // strideY + 1, (in_width - filter_width) // strideX + 1, filter_out_channels))
	for i in range(num_examples):
		for j in range(filter_out_channels):
				for y in range(in_height - filter_height+1):
					for x in range(in_width - filter_width+1):
						in_chan_sum = 0
						for r in range(filter_in_channels):
							in_chan_sum += np.sum(inputs[i, y : y + filter_height, x: x + filter_width, r] * filters[:, :, r, j])
						output[i, y, x, j] = in_chan_sum

	return tf.cast(tf.convert_to_tensor(output), tf.float32)

def same_test_0():
	'''
	Simple test using SAME padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="SAME")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="SAME")
	print("SAME_TEST_0:", "my conv2d:", my_conv[0][0][0], "tf conv2d:", tf_conv[0][0][0].numpy())

def valid_test_0():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_0:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_1():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[3,5,3,3],[5,1,4,5],[2,5,0,1],[3,3,2,1]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = tf.Variable(tf.random.truncated_normal([3, 3, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_2():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[1,3,2,1],[1,3,3,1],[2,1,1,3],[3,2,3,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = np.array([[1,2,3],[0,1,0],[2,1,2]]).reshape((3,3,1,1)).astype(np.float32)
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def main():
	# TODO: Add in any tests you may want to use to view the differences between your and TensorFlow's output
	return

if __name__ == '__main__':
	main()
