import tensorflow as tf
import matplotlib.pyplot as plt
from data_utils import _parse_function



file_names = [ '/home/yasaman/HN/hn_data.tfrecords']
input_files = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(input_files)
dataset = dataset.map(_parse_function)
dataset = dataset.repeat()
dataset = dataset.padded_batch(12, padded_shapes=(tf.TensorShape([None]),
	tf.TensorShape([None]),tf.TensorShape([None]), tf.TensorShape([None]),
		tf.TensorShape([None]),tf.TensorShape([None, 224, 224, 3])))


iterator = dataset.make_initializable_iterator()
next_id, next_gender, next_age, next_label, next_times, next_images = iterator.get_next()


with tf.Session() as sess:
	sess.run(iterator.initializer, feed_dict={input_files:file_names})
	first_id, first_age, first_gender, first_times, first_images = sess.run(
	(next_id, next_age, next_gender, next_times, next_images))
	print("age ... ", first_age, "\n gender ... ", first_gender)
	print("times .. ", first_times, "id ... ", first_id)
	print("image ..", first_images.shape)
	plt.figure()
	plt.imshow(first_images[0,0,:,:,:])
	plt.show()

