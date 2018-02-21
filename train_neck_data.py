#!/usr/bin/python
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import numpy as np
from tensorflow.python.tools import inspect_checkpoint as chkp
VAL_SIZE = 0.2


images = np.load('/home/yasaman/HN/neck_images.npy')
labels = np.load('/home/yasaman/HN/neck_labels_cor.npy')

slim = tf.contrib.slim
vgg = nets.vgg

labels = np.transpose(np.asarray([labels, 1-labels]))
images = np.reshape(images, [-1, 224, 224, 1])
# images are black and white but vgg16 needs 3 channels
images = np.repeat(images, 3, axis=3)
test_idx = np.random.choice(images.shape[0], int(VAL_SIZE * images.shape[0]), replace=False)
test_images = images[test_idx]
test_labels = labels[test_idx]
test_idx.sort()
test_idx = test_idx[::-1]
images = np.delete(images, test_idx, axis=0)
labels = np.delete(labels, test_idx, axis=0)
# removed validation set from training set

print(sum(labels[:,0]), images.shape, "val set", test_images.shape)


in_images = tf.placeholder(images.dtype, images.shape)
in_labels = tf.placeholder(labels.dtype, labels.shape)

val_images = tf.placeholder(test_images.dtype, test_images.shape)
val_labels = tf.placeholder(test_labels.dtype, test_labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((in_images, in_labels))
dataset = dataset.shuffle(6000)
dataset = dataset.batch(64)
dataset = dataset.repeat()
#dataset_it = dataset.make_initializable_iterator()

val_set = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_set = val_set.batch(test_images.shape[0])

iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

next_images, next_labels = iterator.get_next()

logits, intermed = vgg.vgg_16(next_images, num_classes=2)

train_init_op = iterator.make_initializer(dataset)
val_init_op = iterator.make_initializer(val_set)

loss = tf.losses.softmax_cross_entropy(next_labels, logits)
total_loss = tf.losses.get_total_loss()
tf.summary.scalar('xentropy loss', total_loss)


train = tf.train.GradientDescentOptimizer(0.001).minimize(total_loss)


img_net_path = '/home/yasaman/HN/image_net_trained/vgg_16.ckpt'
us_path = '/home/yasaman/HN/neck_us_trained'

#inspecting checkpoint file 
chkp.print_tensors_in_checkpoint_file(img_net_path, tensor_name='',  all_tensors=False, all_tensor_names=True)



# restoring only convolutional layers
scratch_variables = ['vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8']
restored_variables = tf.contrib.framework.get_variables_to_restore(exclude=scratch_variables)

print("restored variables ....", restored_variables)
restorer = tf.train.Saver(restored_variables)
saver = tf.train.Saver()

# for variables that have to be initialized from scratch
detailed_scratch_vars = []
for layer in scratch_variables:
	detailed_scratch_vars.extend(tf.contrib.framework.get_variables(scope=layer))


print("fc layers.....", detailed_scratch_vars)
init_scratch = tf.variables_initializer(detailed_scratch_vars)

with tf.Session() as sess:
	restorer.restore(sess, img_net_path)
	sess.run(init_scratch)
	sess.run(train_init_op, feed_dict={in_images:images,in_labels:labels})
	merged_summaries = tf.summary.merge_all()
	
	writer = tf.summary.FileWriter('/home/yasaman/HN/run_log/', sess.graph)
	for i in range(5000):
		sess.run(train)
		if(i%100 == 0):
			summ = sess.run(merged_summaries)
			writer.add_summary(summ, i)
			print(sess.run(total_loss))
			saver.save(sess, us_path)
	writer.close()





