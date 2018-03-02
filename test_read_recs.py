import tensorflow as tf


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
	#image = tf.reshape(image, [-1, 224, 224, 3])
	times = tf.cast(sequence_parsed['times'], tf.int64)
	base_age = tf.cast(context_parsed['age'], tf.int64)
	gender = tf.cast(context_parsed['gender'], tf.int64)
	label = tf.cast(context_parsed['label'], tf.int64)
	return  gender, base_age, label


file_names = [ '/home/yasaman/HN/hn_data.tfrecords']
dataset = tf.data.TFRecordDataset(file_names)
dataset = dataset.map(_parse_function)
dataset = dataset.repeat()
dataset = dataset.padded_batch(12, padded_shapes=(tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None])))#,tf.TensorShape([None])))
iterator = dataset.make_one_shot_iterator()
next_gender, next_age, next_label = iterator.get_next()
# next_times, next_images,



with tf.Session() as sess:
	#sess.run(iterator.initializer)

 	first_age, first_gender = sess.run((next_age, next_gender))

#	print("gender ...\n", first_gender, "age ...\n", first_age)
