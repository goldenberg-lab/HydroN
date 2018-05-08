import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import numpy as np
from data_utils import _parse_function
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


slim = tf.contrib.slim
vgg = nets.vgg

PATH_CNN_WEIGHTS = '/home/yasaman/HydroN/HN/cnn_neck_weights/neck_us_trained'
PATH_HN_IM = '/home/yasaman/HydroN/HN/serialized_data/last_im_sample.tfrecords'


input_files = tf.placeholder(tf.string, shape=None)


dataset = tf.data.TFRecordDataset(input_files)
dataset = dataset.map(_parse_function)
dataset = dataset.batch(55)


iterator = dataset.make_initializable_iterator()


next_id, _, _, next_label, _, _, _, _, _, _, next_images = iterator.get_next()
next_images = tf.squeeze(next_images, axis=1)
next_label = tf.squeeze(next_label)
next_id = tf.squeeze(next_id)

logits, intermed = vgg.vgg_16(next_images, num_classes=2, is_training=False, spatial_squeeze=False)

fc7 = intermed['vgg_16/fc7']
fc7 = tf.squeeze(fc7)

restorer = tf.train.Saver()


 

with tf.Session() as sess:
	restorer.restore(sess, PATH_CNN_WEIGHTS)
	sess.run(iterator.initializer, feed_dict={input_files:PATH_HN_IM})
	fc7_feat, labels, ids = sess.run((fc7,next_label, next_id))


embedded_feat = TSNE().fit_transform(fc7_feat)
unique_id = np.unique(ids)
cmap = plt.get_cmap('Paired')
id_col = dict(zip(unique_id, cmap(np.linspace(0,1, unique_id.shape[0]))))
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(unique_id.shape[0]):
	if np.sum(labels[ids==unique_id[i]]) > 0:
		ax.plot(embedded_feat[ids==unique_id[i], 0],
			 embedded_feat[ids==unique_id[i], 1],
			'+', label=str(unique_id[i]),c=id_col[unique_id[i]])
	else:
		ax.plot(embedded_feat[ids==unique_id[i], 0],
			 embedded_feat[ids==unique_id[i], 1],
			'o', label=str(unique_id[i]),c=id_col[unique_id[i]])
	

ax.set_title("o --> normal, + --> surgery")
plt.legend()
plt.show()
