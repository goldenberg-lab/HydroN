import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import numpy as np
from data_utils import _parse_function
from GRU_RNN import RNN_GRU, weight_bias
from explore_data_utils import display

BATCH_SIZE = 12
slim = tf.contrib.slim
vgg = nets.vgg
STATE_SIZE = 512


PATH_CNN_WEIGHTS = '/home/yasaman/HN/neck_us_trained'
file_names = ['/home/yasaman/HN/hn_data.tfrecords']

input_files = tf.placeholder(tf.string, shape=None)

dataset = tf.data.TFRecordDataset(input_files)
dataset = dataset.map(_parse_function)
dataset = dataset.repeat()
dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes=(tf.TensorShape([None]),
		tf.TensorShape([None]),tf.TensorShape([None]),
		tf.TensorShape([None]), tf.TensorShape([None]),
		tf.TensorShape([None]), tf.TensorShape([None]),
		tf.TensorShape([None]), tf.TensorShape([None]),
		tf.TensorShape([None]), tf.TensorShape([None, 224, 224, 3])))

iterator = dataset.make_initializable_iterator()
next_id, next_gender, next_age, next_label, next_length, next_circum, next_side, next_etiology, next_vcug, next_times, next_images = iterator.get_next()

all_batch_images = tf.reshape(next_images, [-1, 224, 224, 3])
all_batch_times = tf.reshape(next_times, [BATCH_SIZE, -1, 1])
all_batch_times = tf.cast(all_batch_times, tf.float32)

# prepare context features and feed them to RNN as first state
gender = tf.one_hot(next_gender, depth=2, dtype=tf.int64)
gender = tf.reshape(gender, [-1, 2])
circum = tf.one_hot(next_circum, depth=4, dtype=tf.int64)
circum = tf.reshape(circum, [-1, 4])
side = tf.one_hot(next_side, depth=3, dtype=tf.int64)
side = tf.reshape(side, [-1, 3])
etiology = tf.one_hot(next_etiology, depth=2, dtype=tf.int64)
etiology = tf.reshape(etiology, [-1, 2])
vcug = tf.one_hot(next_vcug, depth=2, dtype=tf.int64)
vcug = tf.reshape(vcug, [-1, 2])
context_feat = tf.concat([gender, next_age, circum, side, etiology, vcug], axis=1)
context_feat = tf.reshape(context_feat, [-1, 14])
num_ctx_feat = int(context_feat.get_shape()[1])
print(int(num_ctx_feat))
with tf.variable_scope('RNN'):
	W_context, _ = weight_bias(num_ctx_feat, STATE_SIZE)

context_feat = tf.cast(context_feat, tf.float32)
initial_state = tf.matmul(context_feat, W_context)

# num classes =2 because finetuned on neck ultrasound images, want to restore all weights, but will not use all of them
logits, intermed = vgg.vgg_16(all_batch_images, num_classes=2)
extr_feat = intermed['vgg_16/fc7']
num_feat = extr_feat.get_shape()[3]
extr_feat = tf.reshape(extr_feat, [BATCH_SIZE,-1, num_feat])
# concatenate time to the end of features from CNN
feat_seq = tf.concat([extr_feat, all_batch_times], axis=2)

# one hot vectors for labels
labels = tf.one_hot(next_label, depth=3)
labels = tf.reshape(labels, [-1, 3])

lengths = tf.reshape(next_length, [BATCH_SIZE])

rnn = RNN_GRU(feat_seq, labels, lengths, init_state=initial_state, num_hidden=STATE_SIZE,  num_layers=1)

# restore CNN weights
all_cnn_var = tf.global_variables(scope='vgg_16')
all_rnn_var = tf.global_variables(scope='RNN')
restorer = tf.train.Saver(all_cnn_var)

init_rnn_var = tf.variables_initializer(all_rnn_var)


with tf.Session() as sess:
	# initialize variables and dataset iterator
	restorer.restore(sess,PATH_CNN_WEIGHTS)
	sess.run(init_rnn_var)
	sess.run(iterator.initializer, feed_dict={input_files:file_names})
	all_images, all_times, first_times = sess.run((all_batch_images, all_batch_times, next_times))
	print("all_times", all_times.shape, "all images shape",
		 all_images.shape, "original batch time  shape ",
		 first_times)
	print(feat_seq.get_shape(), "label shape", labels.get_shape())
	#display(all_images[0:24 ,:,:,:], 4,6)	
	for i in range(1000):
		if (i%100 == 0):
			print(sess.run(rnn.cost))
		sess.run(rnn.optimize)

