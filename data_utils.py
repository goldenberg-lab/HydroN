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


def get_us_times(row):
	''' return age_us for ultrasounds that happened. '''
	times = []
	for i in range(1,10):
		times.append(row['age_us'+str(i)])
	times = [time for time in times if len(time)>0]
	return np.asarray(times)


def read_csv(path):
	''' return a dictionary of subject and their clinical info:
	ID -> (gender, baseline_age, [us_times], label)
	'''
	all_data = dict()
	with open(path) as clinc_data:
		reader = csv.DictReader(clinc_data)
		for example in reader:
			gender = int(example['gender']) - 1
			age_bl = int(example['age_baseline'])
			us_dates = get_us_times(example)	
			label = int(example['surgery1'])
			all_data[example['study_id']] = (gender,
					age_bl, us_dates, label)
	return all_data


def subj_images(image_dir):
	''' read images for this subject, and stack them (axis=3) 
	into numpy array.
	'''
	#print(image_dir)
	assert os.path.isdir(image_dir)
	image_names = os.listdir(os.path.join(image_dir))
	# filter out  non image files 
	image_names = [name  for name in image_names if '.png' in name]
	image_names.sort()	

	num_images = len(image_names)
	
	all_subj_images =  np.zeros((IMG_DIM, IMG_DIM, 3, num_images), dtype=np.float32) 
	for i in range(num_images):
		image = Image.open(os.path.join(image_dir, image_names[i]))
		all_subj_images[:,:,:,i] = _process_image(image)
	return all_subj_images	
		

def _process_image(im):
	''' im is an image object.
	'''
	im = im.resize((IMG_DIM, IMG_DIM), Image.LANCZOS)
	im_arr = np.asarray(im, dtype=np.float32)
	return im_arr[:,:,:3]
	




