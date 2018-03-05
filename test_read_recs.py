import tensorflow as tf
import matplotlib.pyplot as plt

def _parse_function(example_proto):
	context_feature ={
		'id': tf.FixedLenFeature([], tf.int64),
		'gender': tf.FixedLenFeature([], tf.int64),
		'age':  tf.FixedLenFeature([], tf.int64),
		'label': tf.FixedLenFeature([], tf.int64)}
	sequence_feature = {	
		'times': tf.FixedLenSequenceFeature([], tf.int64),
		'images': tf.FixedLenSequenceFeature([], tf.string)}

	context_parsed, sequence_parsed = tf.parse_single_sequence_example(
		example_proto, context_features=context_feature,
		sequence_features=sequence_feature)

	id_no = tf.cast(context_parsed['id'], tf.int64)
	image = tf.decode_raw(sequence_parsed['images'], tf.float32)
	image = tf.reshape(image, [-1, 224, 224, 3])
	times = tf.cast(sequence_parsed['times'], tf.int64)
	base_age = tf.cast(context_parsed['age'], tf.int64)
	gender = tf.cast(context_parsed['gender'], tf.int64)
	label = tf.cast(context_parsed['label'], tf.int64)
	return  (tf.expand_dims(id_no, axis=0),
		tf.expand_dims(gender, axis=0),
		tf.expand_dims(base_age,axis=0),
		tf.expand_dims(label, axis=0),
		times,
		image)



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

