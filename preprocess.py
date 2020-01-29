from PIL import Image, ImageFilter
from os import listdir
from os.path import isfile, join
import numpy
import scipy
import cv2
from scipy.ndimage.filters import gaussian_filter
from skimage import feature
from random import seed
from random import randint

AGE_CLASSES = [ "MTYL_17", "MTYL_28", "MTYL_52", "MTYL_76", "MTYL_100", "MTYL_124", "Po1g_100" ]

SIGMAS = [ 0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0 ]
#SIGMAS = [ 0.3 ]



##################################################################################
#
# function: apply_mask()
#
# Apply a binary image mask representing the cell of interest in the frame
# Value of 0 represents NOT A CELL
# Any Non-Zero value represents CELL.  (usually specified as 255)
#
def apply_mask(raw_image, mask_image):
	raw_array  = numpy.array(raw_image)
	mask_array = numpy.array(mask_image)
	(numrows,numcols) = raw_array.shape

	for c in range(numcols):
		for r in range(numrows):
			if(mask_array[r][c] == 0):
				raw_array[r][c] = 0

	masked_raw = Image.fromarray(raw_array);
	return(masked_raw)



#################################################################################################################
#
# From Adrian Rosebrock
#
# https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
#
#
def auto_canny(img, sigma=0.33):

	# preprocess to smooth out details
	#img = cv2.GaussianBlur(img, (3, 3), 0)

	# compute the median of the single channel pixel intensities
	v = numpy.median(img)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v)/4)
	upper = int(min(255, (1.0 + sigma) * v))
	print("lower=%d,upper=%d" %(lower,upper))
	edged = cv2.Canny(img, lower, upper)
 
	# return the edged image
	return edged


#
# Extract spatial context features by choosing pairs of pixels in the 
# vacinity of the primary pixel being classified and computing the intensity
# difference betweeb that paiir.  Vacinity is a NxN neighboorhood.  Edges are 
# handled by reflecting the out-of-bounds pixels into range.
#
# This is described in:
#	Chisheng Wang, Qiqi Shu, Xinyu Wang, Bo Guo, Peng Liu, Qingquan Li,
#	A random forest classifier based on pixel comparison features for urban LiDAR data,
#	ISPRS Journal of Photogrammetry and Remote Sensing, Volume 148, 2019, Pages 75-86, ISSN 0924-2716,
#

def wang_point(x, y, numrows, numcols, radius):
	
	xr = randint(-radius,radius)
	yr = randint(-radius,radius)

	if( (x+xr >= numcols) or (x+xr < 0) ):
		x1=x-xr
	else:
		x1=x+xr
	if( (y+yr >= numrows) or (y+yr < 0) ):
		y1=y-yr
	else:
		y1=y+yr

	return(x1,y1)




def wang_function(img, radius):

	img_array = numpy.array(img)
	new_array = numpy.ndarray(img_array.shape,dtype=numpy.float32)
	(numrows,numcols) = img_array.shape
	for x in range(numcols):
		for y in range(numrows):
			(x1,y1) = wang_point(x, y, numrows, numcols, radius)
			(x2,y2) = wang_point(x, y, numrows, numcols, radius)

			diff=abs(img_array[y1][x1] - img_array[y2][x2])
			new_array[y][x] = diff

	return(new_array)


#
# Return the size of the feature vector
# This function depends on manually counting the number of appends() in image_preprocess
# multiplying this by the length of the SIGMA parameter list and adding 1 for the original image
# and adding +1 for the binary image
#
def feature_count():
	c = 10
	#wang = 5*5
	wang = 0
	return(len(SIGMAS)*c+len(AGE_CLASSES)+wang+1)	

#
# given an age class label and the filename of the current image
# return 
#
def label_age(img_array, age, filename):
	new_array = numpy.ndarray(img_array.shape, dtype=numpy.uint8)

	if age in filename:
		label = 255
	else:
		label = 0

	new_array.fill(label)

	return(new_array)
	



#
# The structure of this function must match the structure of the image_preprocess() function
# exactly in order for feature labels to match 
#
def feature_labels(sigmas = SIGMAS):

	labels = []

	labels.append("Original")	

	for age in AGE_CLASSES:
		labels.append(age)	

	for s in sigmas:
		# Gaussian Smoothing
		l = "Gaussian_Smoothing_" + str(s)
		labels.append(l)

		# Sobel Edge Detection
		l = "Sobel_Edge_Detection_" + str(s)
		labels.append(l)

		# Laplacian of Gaussian Edge Detection
		l = "Laplacian_of_Gaussian_Edge_Detection_" + str(s)
		labels.append(l)

		# Gaussian Gradient Magnitude Edge Detection
		l = "Gaussian_Gradient_Magnitude_Edge_Detection_" + str(s)
		labels.append(l)

		# Difference of Gaussians
		l = "Difference_of_Gaussians_" + str(s)
		labels.append(l)

		# Structure Tensor Eigenvalues
		l = "Structure_Tensor_Eigenvalues_Large_" + str(s)
		labels.append(l)
		l = "Structure_Tensor_Eigenvalues_Small_" + str(s)
		labels.append(l)

		# Hessian Matrix
		l = "Hessian_Matrix_Hrr_" + str(s)
		labels.append(l)
		l = "Hessian_Matrix_Hrc_" + str(s)
		labels.append(l)
		l = "Hessian_Matrix_Hcc_" + str(s)
		labels.append(l)

	return(labels)



def image_preprocess(filename, original_image, sigmas = SIGMAS):

	x=0

	original_image_array = numpy.array(original_image)
	images = numpy.empty((78, original_image_array.size), dtype=numpy.uint8)

	images[x]=original_image_array.flatten(order='F'); x+=1

	for age in AGE_CLASSES:
		images[x]=label_age(original_image_array, age, filename).flatten(order='F'); x+=1

	for s in sigmas:

		# Gaussian Smoothing
		images[x] = gaussian_filter(original_image_array, sigma=s).flatten(order='F'); x+=1

		# Sobel Edge Detection
		images[x] = scipy.ndimage.sobel(original_image_array, mode='constant', cval=s).flatten(order='F'); x+=1

		# Laplacian of Gaussian Edge Detection
		images[x] = scipy.ndimage.gaussian_laplace(original_image_array, sigma=s).flatten(order='F'); x+=1

		# Gaussian Gradient Magnitude Edge Detection
		images[x] = scipy.ndimage.gaussian_gradient_magnitude(original_image_array, sigma=s).flatten(order='F'); x+=1

		# Difference of Gaussians
		k = 1.7
		tmp1_array = scipy.ndimage.gaussian_filter(original_image_array, sigma=s*k) 
		tmp2_array = scipy.ndimage.gaussian_filter(original_image_array, sigma=s)
		images[x] = (tmp1_array - tmp2_array).flatten(order='F'); x+=1
		
		# Structure Tensor Eigenvalues
		Axx,Axy,Ayy = feature.structure_tensor(original_image, sigma=s)
		large_array,small_array = feature.structure_tensor_eigvals(Axx,Axy,Ayy)
		images[x] = large_array.flatten(order='F'); x+=1
		images[x] = small_array.flatten(order='F'); x+=1

		# Hessian Matrix
		Hrr,Hrc,Hcc = feature.hessian_matrix(original_image, sigma=s, order='rc')
		images[x]=Hrr.flatten(order='F'); x+=1
		images[x]=Hrc.flatten(order='F'); x+=1
		images[x]=Hcc.flatten(order='F'); x+=1

	#for i in range(5):
		#image_list.append(wang_function(original_image, 1))
		#image_list.append(wang_function(original_image, 2))
		#image_list.append(wang_function(original_image, 3))
		#image_list.append(wang_function(original_image, 4))
		#image_list.append(wang_function(original_image, 5))

	images=numpy.transpose(images,(1,0))

	#print("preprocess")
	#print(type(images))
	#print(images.shape)

	return(images)


def output_preprocessed(images, dir):
	index = 0
	for f in images:
		pathname = dir + "/" + str(index) + ".tif"
		x = Image.fromarray(f)
		x.save(pathname, "TIFF")
		index = index + 1
	
