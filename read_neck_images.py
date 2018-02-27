import numpy as np
import os
from PIL import Image
import os.path

IM_DIR = '/home/yasaman/HN/neck_train'
SUB_IMG_PATH = '/home/yasaman/HN/subj_img.npy'
LABEL_PATH = '/home/yasaman/HN/neck_label.npy'

image_list = os.listdir(IM_DIR)
subj_img = np.load(SUB_IMG_PATH)
labels_origin = np.load(LABEL_PATH)


print(subj_img)

images = []
labels = []

for i in range(subj_img.shape[0]):
	us_im = Image.open(os.path.join(IM_DIR, str(subj_img[i]).lstrip('b').replace("'", '') + ".tif"))
	us_im = us_im.resize((224, 224), Image.LANCZOS)
	us_array = np.asarray(us_im, dtype='float32')
	images.append(us_array)
	labels.append(labels_origin[i])
	if (i%500 == 0): print(i, subj_img[i]) 

images = np.asarray(images)
np.save("neck_images", images)
np.save("neck_labels_cor", labels)




