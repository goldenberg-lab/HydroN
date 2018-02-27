import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from GRU_RNN import RNN_GRU
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)


inputs = tf.placeholder(tf.float32, [None, 28, 28])
targets = tf.placeholder(tf.float32, [None, 10])

rnn = RNN_GRU(inputs, targets, num_hidden=512, num_layers=2)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())	
	for i in range(100):
		image_batch, label_batch = mnist.train.next_batch(128)
		image_batch = np.reshape(image_batch, [-1, 28, 28])
#		length = sess.run(rnn.length, feed_dict={inputs:image_batch, targets:label_batch})
#		print("length ...",length.shape)	
		print(sess.run(rnn.cost, feed_dict={inputs:image_batch, targets:label_batch}))
		
#		print("Logits ....",sess.run(rnn.logits, feed_dict={inputs:image_batch, targets:label_batch}))
#		print("weights ....",sess.run(rnn._weight))
#		print("rnn output ....",sess.run(rnn._last, feed_dict={inputs:image_batch, targets:label_batch}))
		sess.run(rnn.optimize, feed_dict={inputs:image_batch, targets:label_batch})
