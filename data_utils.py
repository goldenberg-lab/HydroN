import numpy as np
from PIL import Image
import csv
import os
import os.path
import csv
import tensorflow as tf
from explore_data_utils import display
import scipy.ndimage
from skimage.transform import resize
from skimage import img_as_float, img_as_ubyte

IMG_DIM = 224

def _parse_function(example_proto):
	context_feature ={
		'id': tf.FixedLenFeature([], tf.int64),
		'gender': tf.FixedLenFeature([], tf.int64),
		'age':  tf.FixedLenFeature([], tf.int64),
		'label': tf.FixedLenFeature([], tf.int64),
		'len': tf.FixedLenFeature([], tf.int64),
		'circum': tf.FixedLenFeature([], tf.int64),
		'side': tf.FixedLenFeature([], tf.int64),
		'etiology': tf.FixedLenFeature([], tf.int64),
		'vcug': tf.FixedLenFeature([], tf.int64)}
	sequence_feature = {	
		'times': tf.FixedLenSequenceFeature([], tf.int64),
		'images': tf.FixedLenSequenceFeature([], tf.string)}

	context_parsed, sequence_parsed = tf.parse_single_sequence_example(
		example_proto, context_features=context_feature,
		sequence_features=sequence_feature)

	id_no = tf.cast(context_parsed['id'], tf.int64)
	image = tf.decode_raw(sequence_parsed['images'], tf.float32)
	image = tf.reshape(image, [-1, IMG_DIM, IMG_DIM, 3])
	times = tf.cast(sequence_parsed['times'], tf.int64)
	base_age = tf.cast(context_parsed['age'], tf.int64)
	gender = tf.cast(context_parsed['gender'], tf.int64)
	label = tf.cast(context_parsed['label'], tf.int64)
	circum = tf.cast(context_parsed['circum'], tf.int64)
	side =  tf.cast(context_parsed['side'], tf.int64)
	etiology =  tf.cast(context_parsed['etiology'], tf.int64)
	vcug =  tf.cast(context_parsed['vcug'], tf.int64)
	length = tf.cast(context_parsed['len'], tf.int32)
	return  (tf.expand_dims(id_no, axis=0),
		tf.expand_dims(gender, axis=0),
		tf.expand_dims(base_age,axis=0),
		tf.expand_dims(label, axis=0),
		tf.expand_dims(length, axis=0),
		tf.expand_dims(circum, axis=0),
		tf.expand_dims(side, axis=0),
		tf.expand_dims(etiology, axis=0),
		tf.expand_dims(vcug, axis=0),
		times,
		image)


def order_match(images, times):
	''' given list of images and their corresponding time stamps, return 
	array of images sorted in axis 0 by time, and ordered list of time
	differences.
	'''
	ordered_images = np.zeros(images.shape, dtype=np.float32)
	idx_sorted = sorted(range(len(times)), key= lambda i: times[i])
	ordered_times = [times[i] for i in idx_sorted]	
	for i in range(len(idx_sorted)):
		ordered_images[i,:,:,:] = images[idx_sorted[i],:,:,:]
	# take difference
	for i in range(len(ordered_times)-1,0, -1):
		ordered_times[i] = ordered_times[i] - ordered_times[i-1]
	return ordered_images, ordered_times


def make_seq_example(id_no, info, images):
	''' return a tf.train.SequenceExample instance for single example '''
	# check all images have a time
	if (images.shape[0] > len(info['times'])):
		print('subj no ', id_no, 'num images: ', images.shape[0],
			'num time stamps ', len(info['times']))
		raise TypeError('There are more images than time stamps, some image must be missing a time stamp')
	# if there are more time stamps take the ones that have an image
	info['times'] = info['times'][0:images.shape[0]]
	images, times = order_match(images, info['times']) 
	example = tf.train.SequenceExample()
	# non sequential features
	example.context.feature['gender'].int64_list.value.append(info['gender'])	
	example.context.feature['age'].int64_list.value.append(info['age_baseline'])	
	example.context.feature['label'].int64_list.value.append(info['label'])
	example.context.feature['id'].int64_list.value.append(int(id_no))
	example.context.feature['len'].int64_list.value.append(len(times))
	example.context.feature['circum'].int64_list.value.append(info['circum'])
	example.context.feature['side'].int64_list.value.append(info['side'])
	example.context.feature['etiology'].int64_list.value.append(info['etiology'])
	example.context.feature['vcug'].int64_list.value.append(info['vcug'])


	# two sequential featuesi
	images_fl = example.feature_lists.feature_list['images']
	times_fl = example.feature_lists.feature_list['times']
	for i in range(len(info['times'])):
		images_fl.feature.add().bytes_list.value.append(images[i].tostring())
		times_fl.feature.add().int64_list.value.append(times[i])
	return example
	

def get_us_times(row):
	''' return age_us for ultrasounds that happened. '''
	times = []
	for i in range(1,10):
		times.append(row['age_us'+str(i)])
	times = [int(time) for time in times if len(time)>0]
	return np.asarray(times)


def get_circ_stat(status):
	''' replace circumcision status for missing.
	four possible states: yes,no, n/a, missing.  '''
	if (status == ''): status = 3
	return int(status)


def get_label(example):
	''' if no surgery(1) check for UTI(2), else healthy(0)
	'''
	label = int(example['surgery1'])
	if not label:
		label = int(example['uti1'])
		# if it is 1 change it to 2, otherwise stays 0
		label = label * 2
	return label


def read_csv(path):
	''' return a dictionary of subject and their clinical info:
	ID -> {gender: int, baseline_age:int, us_times: [int], label:[int]}
	'''
	all_data = dict()
	with open(path) as clinc_data:
		reader = csv.DictReader(clinc_data)
		for example in reader:
			gender = int(example['gender']) - 1
			age_bl = int(example['age_baseline'])
			us_dates = get_us_times(example)
			circum = get_circ_stat(example['circumcision_status'])	
			label = get_label(example)
			surgery = int(example['surgery1'])
			side = int(example['laterality']) - 1
			etiology = int(example['etiology']) - 1
			vcug = int(example['vcug1'])
			all_data[example['study_id']] =  {'gender':gender,
				'age_baseline':age_bl, 'circum':circum,
				'times':us_dates, 'label':label, 'side':side,
				'etiology':etiology, 'vcug':vcug, 'surgery':surgery}
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
		all_subj_images[i,:,:,:] = _process_image(image, 6)
	return all_subj_images	

	
def _process_image(im, sigma):
	''' im is an image object.
	'''
	im_arr = np.asarray(im, dtype=np.ubyte)
	# ignore alpha channel
	image = im_arr[:,:,:3]
	image = img_as_float(image)
	#image = image - image.mean()
	#image = image - image.std()
	image = scipy.ndimage.filters.gaussian_filter(image, sigma)
	image = resize(image, (IMG_DIM, IMG_DIM))

	return image
	


