import functools 
import tensorflow as tf
import numpy as np

# this is based on https://danijar.com/variable-sequence-lengths-in-tensorflow/

def scoped_property(func):
	attribute = '_cache_' + func.__name__

	@property
	@functools.wraps(func)
	def decorator(self):
		if not hasattr(self, attribute):
			with tf.variable_scope(func.__name__):
				setattr(self, attribute, func(self))
		return getattr(self, attribute)
	return decorator


def weight_bias(in_size, out_size):
	weight = tf.truncated_normal([in_size, out_size], stddev=0.1)
	bias = tf.constant(0.0, shape=[out_size])
	return tf.Variable(weight), tf.Variable(bias)


class RNN_GRU():

	def __init__(self, features, decision,
		 num_hidden=512, num_layers=1):
		self.inputs = features
		self.target = decision
		self._num_hidden = num_hidden
		self._num_layers = num_layers
		self.length
		self.logits
		self.cost
		self.optimize


	@scoped_property
	def length(self):
		used = tf.sign(tf.reduce_max(tf.abs(self.inputs), reduction_indices=2))
		length = tf.reduce_sum(used, reduction_indices=1)
		length = tf.cast(length, tf.int32)
		return length


	@scoped_property
	def logits(self):
		# default tanh activation
		layers = [tf.nn.rnn_cell.LSTMCell(self._num_hidden) for i in range(self._num_layers)]
		multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(layers)
		outputs , final_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
				 inputs=self.inputs,
				 dtype=tf.float32,
				sequence_length=self.length)
		## softmax
		num_classes = int(self.target.get_shape()[1])
		# last output 
		outputs = tf.transpose(outputs, [1, 0, 2])
		last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
		weight, bias = weight_bias(self._num_hidden,num_classes)
		logits = tf.matmul(last, weight) + bias
		return logits


	@scoped_property
	def cost(self):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target, logits=self.logits))
		return cross_entropy


	@scoped_property
	def optimize(self):
		optimizer = tf.train.AdagradOptimizer(0.01)
		return optimizer.minimize(self.cost)


	

		







