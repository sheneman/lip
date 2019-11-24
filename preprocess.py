from PIL import Image, ImageFilter
from os import listdir
from os.path import isfile, join
import numpy
import scipy
from scipy.ndimage.filters import gaussian_filter
from skimage import feature

SIGMAS = [ 0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0 ]
#SIGMAS = [ 0.3 ]


#
# Return the size of the feature vector
# This function depends on manually counting the number of appends() in image_preprocess
# multiplying this by the length of the SIGMA parameter list and adding 1 for the original image
# and adding +1 for the binary image
#
def feature_count():
	c = 10
	return(len(SIGMAS)*c+1)	


def image_preprocess(original_image, sigmas = SIGMAS):

	image_list = []
	original_image_array = numpy.array(original_image)
	image_list.append(original_image_array);

	for s in sigmas:

		# Gaussian Smoothing
		img2_array = gaussian_filter(original_image_array, sigma=s)
		image_list.append(img2_array)

		# Sobel Edge Detection
		img2_array = scipy.ndimage.sobel(original_image_array, mode='constant', cval=s)
		image_list.append(img2_array)

		# Laplacian of Gaussian Edge Detection
		img2_array = scipy.ndimage.gaussian_laplace(original_image_array, sigma=s) 
		image_list.append(img2_array)

		# Gaussian Gradient Magnitude Edge Detection
		img2_array = scipy.ndimage.gaussian_gradient_magnitude(original_image_array, sigma=s) 
		image_list.append(img2_array)

		# Difference of Gaussians
		k = 1.7
		tmp1_array = scipy.ndimage.gaussian_filter(original_image_array, sigma=s*k) 
		tmp2_array = scipy.ndimage.gaussian_filter(original_image_array, sigma=s)
		dog = tmp1_array - tmp2_array
		image_list.append(dog)

		# Structure Tensor Eigenvalues
		Axx,Axy,Ayy = feature.structure_tensor(original_image, sigma=s)
		large_array,small_array = feature.structure_tensor_eigvals(Axx,Axy,Ayy)
		image_list.append(large_array)
		image_list.append(small_array)

		# Hessian Matrix
		Hrr,Hrc,Hcc = feature.hessian_matrix(original_image, sigma=s, order='rc')
		image_list.append(Hrr)
		image_list.append(Hrc)
		image_list.append(Hcc)

	return(image_list)



def output_preprocessed(images, dir):
	index = 0
	for f in images:
		pathname = dir + "/" + str(index) + ".tif"
		x = Image.fromarray(f)
		x.save(pathname, "TIFF")
		index = index + 1
	
