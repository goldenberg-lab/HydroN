import numpy as np
from PIL import Image
import csv
import os
import os.path
import csv
import tensorflow as tf


IMG_DIM = 224

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_seq_example(id_no, info, label_map):
	''' return a tf.train.SequenceExample instance for single example '''
	# get images 
	path_to_subj = os.path.join(label_map[info['label']], id_no)
	images = subj_images(path_to_subj)
	# check all images have a time
	if (images.shape[0] > len(info['times'])):
		print('subj no ', id_no, 'num images: ', images.shape[0],
			'num time stamps ', len(info['times']))
		raise TypeError('There are more images than time stamps, some image must be missing a time stamp')
	# if there are more time stamps take the ones that have an image
	info['times'] = info['times'][0:images.shape[0]]
	example = tf.train.SequenceExample()
	# non sequential features
	example.context.feature['gender'].int64_list.value.append(info['gender'])	
	
	example.context.feature['age'].int64_list.value.append(info['age_baseline'])	
	
	example.context.feature['label'].int64_list.value.append(info['label'])
	
	example.context.feature['id'].int64_list.value.append(int(id_no))
	# two sequential featues
	images_fl = example.feature_lists.feature_list['images']
	times_fl = example.feature_lists.feature_list['times']
	for i in range(len(info['times'])):
		images_fl.feature.add().bytes_list.value.append(images[i].tostring())
		times_fl.feature.add().int64_list.value.append(info['times'][i])
	return example
	
	


def get_us_times(row):
	''' return age_us for ultrasounds that happened. '''
	times = []
	for i in range(1,10):
		times.append(row['age_us'+str(i)])
	times = [int(time) for time in times if len(time)>0]
	return np.asarray(times)


def read_csv(path):
	''' return a dictionary of subject and their clinical info:
	ID -> {gender: int, baseline_age:int, [us_times]: [int], label:[int]}
	'''
	all_data = dict()
	with open(path) as clinc_data:
		reader = csv.DictReader(clinc_data)
		for example in reader:
			gender = int(example['gender']) - 1
			age_bl = int(example['age_baseline'])
			us_dates = get_us_times(example)	
			label = int(example['surgery1'])
			all_data[example['study_id']] =  {'gender':gender,
				'age_baseline':age_bl, 'times':us_dates,
				 'label':label}
	return all_data


def subj_images(image_dir):
	''' read images for this subject, and stack them (axis=0) 
	into numpy array.
	'''
	#print(image_dir)
	assert os.path.isdir(image_dir)
	image_names = os.listdir(os.path.join(image_dir))
	# filter out  non image files 
	image_names = [name  for name in image_names if '.png' in name]
	image_names.sort()	

	num_images = len(image_names)
	
	all_subj_images =  np.zeros((num_images, IMG_DIM, IMG_DIM, 3), dtype=np.float32) 
	for i in range(num_images):
		image = Image.open(os.path.join(image_dir, image_names[i]))
		all_subj_images[i,:,:,:] = _process_image(image)
	return all_subj_images	
		

def _process_image(im):
	''' im is an image object.
	'''
	im = im.resize((IMG_DIM, IMG_DIM), Image.LANCZOS)
	im_arr = np.asarray(im, dtype=np.float32)
	return im_arr[:,:,:3]
	




