import numpy as np
import pydicom
from pydicom.filereader import read_dicomdir
import os
import os.path
import matplotlib.pyplot as plt
import pytesseract as pytess
from PIL import Image
from skimage.transform import resize
from skimage import img_as_float, exposure
import scipy.ndimage



def read_dicom_dir(dirc):
	''' read all dicom files in this directory, return 
	cropped version of the images, and label for image view: 
	RT/LT SAG, for now images with other labels are ignored.
	'''
	tessdata_dir = '--tessdata-dir "/hpf/tools/centos6/tesseract/3.04.00/share/tesseract"'

	dicom_files = os.listdir(dirc)
	# filter out ones that don't end in .dcm
	dicom_files = list(filter(lambda x: '.dcm' in x, dicom_files))
	images = []
	labels = []
	for dcm_file in dicom_files:
		dataset = pydicom.dcmread(os.path.join(dirc, dcm_file))
		im = dataset.pixel_array
		contrasted_im = exposure.adjust_gamma(im, 2)
		text = pytess.image_to_string(contrasted_im, config=tessdata_dir)
		label = ''
#		print("__________\n",text)
		if (('LONG' in text) or ('SAG' in text)):
			if ('RT' in text):
				label+= 'SAG_RT'
			elif ('LT' in text):
				label+= 'SAG_LT'
		if label:
			im = im[50:-50, 150:-150]
			images.append(im)
			labels.append(label)
	return images,labels


def get_dicom_dir(parent_dir, max_depth=5):
	'''
	Find path to parent directory of dicom images, where the structure 
	is parent_dir/subdir1/subdir2/.../dicom_dir. There is atleast one
	subdir.
	'''
	if max_depth == 0:
		print("max depth reached, no dicoms yet ", parent_dir)
		return 
	sub_dirs = os.listdir(parent_dir)
	sub_path = os.path.join(parent_dir, sub_dirs[0])
#	print(sub_path)
	if  os.path.isdir(sub_path):
		return get_dicom_dir(sub_path, max_depth-1)

	# this is a dumb check but it should work for now
	elif '.dcm' in sub_path:
		return parent_dir
	print("Non dicom files in ", parent_dir)
	return 
	
	

def preprocess_dicom(image):
	'''
	given np array of image, if it has only one channel, copy
	it across channels so that it has three channels.
	filter and resize to 224, and change type of image into float.
	'''
	if ((len(image.shape) < 3) or (image.shape[-1] == 1)):
		image = np.reshape(image, [image.shape[0], image.shape[1], 1])
		image = np.repeat(image, 3, axis=2)
	
	image = img_as_float(image)
	image = scipy.ndimage.filters.gaussian_filter(image, 3)
	image = resize(image, (224, 224))
	return image



if __name__=="__main__":
	
	session_dir_path = '/home/yasaman/HydroN/Hydronephrosis/D4_1/'
	dicom_dir = get_dicom_dir(session_dir_path)
	
	images, labels = read_dicom_dir(dicom_dir)
	print(labels)
	
	

	
