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
			with tf.variable_scope('RNN'):
				with tf.variable_scope(func.__name__):
					setattr(self, attribute, func(self))
		return getattr(self, attribute)
	return decorator


def weight_bias(in_size, out_size):
	weight = tf.truncated_normal([in_size, out_size], stddev=0.1)
	bias = tf.constant(0.0, shape=[out_size])
	return tf.Variable(weight), tf.Variable(bias)


class RNN_GRU():

	def __init__(self, features, decision, lengths,init_state=None,
		num_hidden=512, num_layers=1):
		self.inputs = features
		self.target = decision
		self._num_hidden = num_hidden
		self._init_state = init_state
		self._num_layers = num_layers
		self._length = lengths
		self._bat_size = tf.shape(decision)[0]
		self.logits
		self.cost
		self.optimize


	@scoped_property
	def rnn_out(self):
		# default tanh activation
		layers = [tf.nn.rnn_cell.GRUCell(self._num_hidden) for i in range(self._num_layers)]
		multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(layers)
		init_states = [self._init_state]
		for i in range(1,self._num_layers):
			init_states.append(layers[i].zero_state(self._bat_size))
		outputs , final_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
				inputs=self.inputs,
				dtype=tf.float32,
				sequence_length=self._length,
				initial_state=tuple(init_states))
		return outputs
		

	@scoped_property
	def logits(self):			
		## gather last output in time for softmax
		num_classes = int(self.target.get_shape()[1])
		max_seq_len = tf.shape(self.rnn_out)[1]
		# note output size = num_hidden
		flat_out = tf.reshape(self.rnn_out, [-1, self._num_hidden])	
		# index into different element of the batch
		flat_last_indices = (tf.range(0, self._bat_size) * max_seq_len
		) + (self._length -1)
		self._last = tf.gather(flat_out, flat_last_indices)
		self._weight, self._bias = weight_bias(self._num_hidden,num_classes)
		logits = tf.matmul(self._last, self._weight) + self._bias
		return logits


	@scoped_property
	def cost(self):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target, logits=self.logits))
		return cross_entropy


	@scoped_property
	def optimize(self):
		optimizer = tf.train.AdagradOptimizer(0.001)
		return optimizer.minimize(self.cost)


	

		







