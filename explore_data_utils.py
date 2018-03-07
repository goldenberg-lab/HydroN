import matplotlib.pyplot as plt
import numpy as np



def display(images, num_row, num_column, canvas_ret=False):
	''' column major order.'''
	image_dim = images.shape[1]
	image_channels = images.shape[-1]
	canvas = np.empty((image_dim * num_row, image_dim * num_column, image_channels))
	for i in range(num_row):
                for j in range(num_column):
                        canvas[i*image_dim:(i+1)*image_dim, j*image_dim: (j + 1)*image_dim, :] = images[i*num_column + j]
	if canvas_ret:
		return canvas
	plt.figure()
	plt.imshow(canvas, origin="upper",interpolation='none', cmap=plt.get_cmap('gray'))
	plt.tight_layout()
	plt.show()
