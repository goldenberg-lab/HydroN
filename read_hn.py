import tensorflow as tf
import numpy as np
import os
import os.path
from data_utils import subj_images, read_csv, make_seq_example



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
		example = make_seq_example(study_no, info, label_map) 	
		writer.write(example.SerializeToString())



