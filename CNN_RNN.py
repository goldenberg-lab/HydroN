import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import numpy as np
from data_utils import _parse_function
from GRU_RNN import RNN_GRU
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from explore_data_utils import display

BATCH_SIZE = 12
slim = tf.contrib.slim
vgg = nets.vgg



PATH_CNN_WEIGHTS = '/home/yasaman/HN/neck_us_trained'
file_names = ['/home/yasaman/HN/hn_data.tfrecords']

input_files = tf.placeholder(tf.string, shape=None)

dataset = tf.data.TFRecordDataset(input_files)
dataset = dataset.map(_parse_function)
dataset = dataset.repeat()
dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes=(tf.TensorShape([None]),
	tf.TensorShape([None]),tf.TensorShape([None]), tf.TensorShape([None]),
		tf.TensorShape([None]),tf.TensorShape([None, 224, 224, 3])))

iterator = dataset.make_initializable_iterator()
next_id, next_gender, next_age, next_label, next_times, next_images = iterator.get_next()

all_batch_images = tf.reshape(next_images, [-1, 224, 224, 3])
all_batch_times = tf.reshape(next_times, [BATCH_SIZE, -1, 1])
all_batch_times = tf.cast(all_batch_times, tf.float32)


# num classes =2 because finetuned on neck ultrasound images, want to restore all weights, but will not use all of them
logits, intermed = vgg.vgg_16(all_batch_images, num_classes=2)
extr_feat = intermed['vgg_16/fc7']
num_feat = extr_feat.get_shape()[3]
extr_feat = tf.reshape(extr_feat, [BATCH_SIZE,-1, num_feat])
# concatenate time to the end of features from CNN
feat_seq = tf.concat([extr_feat, all_batch_times], axis=2)

# one hot vectors for labels
labels = tf.one_hot(next_label, depth=2)
labels = tf.reshape(labels, [-1, 2])

rnn = RNN_GRU(feat_seq, labels, num_layers=2)

# restore CNN weights
all_cnn_var = tf.global_variables(scope='vgg_16')
all_rnn_var = tf.global_variables(scope='RNN')
restorer = tf.train.Saver(all_cnn_var)

init_rnn_var = tf.variables_initializer(all_rnn_var)

#print_tensors_in_checkpoint_file(PATH_CNN_WEIGHTS, tensor_name='', all_tensors=False, all_tensor_names=True)

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

#	print("CNN variables ....", all_cnn_var) 
#	print("RNN variables ... ", all_rnn_var)
