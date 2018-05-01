import numpy as np
import pydicom
from pydicom.filereader import read_dicomdir
import os
import os.path
import matplotlib.pyplot as plt
from pytesser import *
from PIL import Image



dicom_dir_path = '/home/yasaman/HydroN/Hydronephrosis/D10_1/download20180426100433/US_2354673614/35198126/70087942'
dicom_files = os.listdir(dicom_dir_path)

dataset = pydicom.dcmread(os.path.join(dicom_dir_path, dicom_files[2]))

print(dataset)
im = dataset.pixel_array
image = Image.fromarray(im)

im = im[50:-50, 150:-150]
plt.figure()
plt.imshow(im)
plt.show()


