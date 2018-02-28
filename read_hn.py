import tensorflow as tf
import numpy as np
import os
import os.path
from data_utils import subj_images, read_csv, _int64_feature, _bytes_feature



BASE_PATH = '/home/yasaman/HN/HN-sample-to-start-with-otherdata-imgs'
SURG_PATH = 'Images/Pyeloplasty/'
CONS_PATH = 'Images/No Surgery/'
CLINC_PATH = 'RAW_PHN_DATA_Sample.csv'


SURG_PATH = os.path.join(BASE_PATH, SURG_PATH)
CONS_PATH = os.path.join(BASE_PATH, CONS_PATH)
surgery_names = os.listdir(SURG_PATH)
surgery_names.sort()
# path to save hydronephrosis
TFREC_HN_PATH = '/home/yasaman/HN/hn_data.tfrecords'

label_map = dict({0:CONS_PATH, 1:SURG_PATH})

all_examples = read_csv(os.path.join(BASE_PATH, CLINC_PATH))

with tf.python_io.TFRecordWriter(TFREC_HN_PATH) as writer:
	for study_no,info in all_examples.items():
		# info[3] label 
		# TODO map these 
		path_to_subj = os.path.join(label_map[info[3]], study_no)
		images = subj_images(path_to_subj)
		images_raw = images.tostring()
		feature ={
			'id': _int64_feature(int(study_no)),
			'gender': _int64_feature(info[0]),
			'base_age': _int64_feature(info[1]),
			'times': _bytes_feature(info[2].tostring()),
			'image': _bytes_feature(images_raw),
			'label':_int64_feature(info[3])}
		example = tf.train.Example(features=tf.train.Features(feature=feature))
		writer.write(example.SerializeToString())



