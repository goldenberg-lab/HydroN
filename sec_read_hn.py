import os
from data_utils import *
import random
import numpy as np
from read_dicom import *


clin_path = '/home/yasaman/HydroN/PHN_datasets/mar_21_raw.csv'
image_path = '/home/yasaman/HydroN/Hydronephrosis'


def filter_laterality(images, labels, side):
	''' weed out images from healthy side, and preprocess images to match 
	shape, and have float type.
	'''
	if side == 2:
		return [preprocess_dicom(image) for image in images]
	elif side == 0:
		# left side only
		images = [preprocess_dicom(images[i]) for i in range(len(images)) if 'LT' in labels[i]]
	elif side == 1:
		# right side only
		images = [preprocess_dicom(images[i]) for i in range(len(images)) if 'RT' in labels[i]]
	else:
		print("invalid laterality value")
	return images


def get_images_id(id_no,id_images, clin_data, image_path):
	''' return a list of 5 series of images.
	Since for each time point, we have multiple images,
	we can generate multiple time series images.
	Each element of the list is a numpy array of shape
	(len time series, image height, image width, num channels=3)
	'''
	# read all images from each time point
	seqs = [[] for i in range(5)]
	id_images[id_no].sort()
	for seq_num in id_images[id_no]:
		dir_name = 'D' + id_no + '_' + seq_num
		path_image_time = get_dicom_dir(os.path.join(image_path,
					dir_name))
		images_time, labels = read_dicom_dir(path_image_time)
		images = filter_laterality(images_time,
			labels, clin_data['side'])
		# sample with replacement
		if(len(images) == 0):
			print("no Sag images found for ", dir_name)
			continue
		random_subset_time = [random.choice(images) for seq in seqs]
		for i in range(len(seqs)):
			seqs[i].append(random_subset_time[i])
		
	seqs = [np.stack(seq, axis=0) for seq in seqs]
	return seqs
		

image_dirs = os.listdir(image_path)
ids = set([name.strip('D').split('_')[0] for name in image_dirs])



clin_data = read_csv(clin_path)
# id_images keeps track of time points where images are present 
id_times = [(name.strip('D').split('_')[0], name.strip('D').split('_')[1]) for name in image_dirs]
id_images = dict()
for stud_id in ids:
	id_images[stud_id] = []
for id_time in id_times:
	id_images[id_time[0]].append(id_time[1])


PATH_TFREC = '/home/yasaman/HydroN/HN/serialized_data/last_im_sample.tfrecords'
writer = tf.python_io.TFRecordWriter(PATH_TFREC)


id_data = dict()
for stud_id in ids:
	id_data[stud_id] = clin_data[stud_id]
	# include only times for which we have images
	id_data[stud_id]['times'] = [id_data[stud_id]['times'][int(time)-1] for time in id_images[stud_id]]
	id_data[stud_id]['images'] = get_images_id(stud_id, id_images,
			id_data[stud_id], image_path)
	for image_seq in id_data[stud_id]['images']:
		example = make_seq_example(stud_id, id_data[stud_id], image_seq[-1:])
	
		writer.write(example.SerializeToString())
		print("writing id no", stud_id)
writer.close()




