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
	edged = cv2.Canny(img, lower, upper)
 
	# return the edged image
	return edged


#################################################################################################################
#
# set pixel to extreme value based on threshhold and comparison of median neighborhood intensity relative 
# to median intensity of overall image
#
#
def median_hood(img_array, radius=4, factor=1.0):


	#img=Image.fromarray(img_array)
	#img.save("debug/ORIG_NEIGH.tif", "TIFF")

	# if the input image is an unsigned 8-bit int, this function may generate an overflow
	#   warning, so skip wang on such images
	if(img_array[0][0].dtype == numpy.uint8):
		return(img_array)

	new_array = numpy.ndarray(img_array.shape,dtype=numpy.float32)
	(numrows,numcols) = img_array.shape
	for x in range(numcols):
		for y in range(numrows):

			x1 = x-radius
			y1 = y-radius

			x2 = x+radius
			y2 = y+radius

			if(x1<0):
				x1=0
			if(y1<0):
				y1=0
			if(x2>=numcols):
				x2=numcols-1
			if(y2>=numrows):
				y2=numrows-1

			neighborhood=img_array[y1:y2, x1:x2]
			if(numpy.median(neighborhood) >= factor*numpy.median(img_array)):
				new_array[y,x] = 1.0
			else:
				new_array[y,x] = -1.0

	#img=Image.fromarray(new_array)
	#img.save("debug/NEW_NEIGH.tif", "TIFF")

	return(new_array)




###############################################################################################
#
# Relative proximity to center of image, scaled from -1 to 1
# 
def center_proximity(img_array):

	new_array = numpy.ndarray(img_array.shape,dtype=numpy.float32)
	(numrows,numcols) = img_array.shape
	for x in range(numcols):
		for y in range(numrows):

			distx = abs((float(x)/float(numcols)-0.5)*2.0)
			disty = abs((float(y)/float(numrows)-0.5)*2.0)
			final = 1.0-((distx+disty)/2.0)

			new_array[y,x] = final

	#img=Image.fromarray(new_array)
	#img.save("debug/NEW_PROX.tif", "TIFF")

	return(new_array)



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

	# if the image is so small that we're still out of bounds, just punt
	if( (x1>=numcols) or (x1<0) ):
		x1 = x
	if( (y1>=numrows) or (y1<0) ):
		y1 = y
		

	return(x1,y1)




def wang_function(img_array, radius):

	# if the input image is an unsigned 8-bit int, this function may generate an overflow
	#   warning, so skip wang on such images
	if(img_array[0][0].dtype == numpy.uint8):
		return(img_array)

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
def feature_count(feature_labels):
	return(len(feature_labels))

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

	labels.append("Wang_1")
	labels.append("Wang_3")
	labels.append("Wang_5")
	labels.append("Wang_7")

	labels.append("median_hood_1.0")
	labels.append("median_hood_1.25")
	labels.append("median_hood_1.5")

	labels.append("center_proximity")

	return(labels)


def simage(img_array, filename):
		
	img=Image.fromarray(img_array)
	f = "./debug/32bit/" + filename
	img.save(f, "TIFF")

	if(filename.startswith('AGE_')):		
		f = "./debug/8bit/" + filename
		img.save(f, "TIFF")
	else:	
		a = cv2.normalize(img_array,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		img=Image.fromarray(a)
		f = "./debug/8bit/" + filename
		img.save(f, "TIFF")

	return(0)
	

def image_preprocess(filename, nfeatures, original_image_array, sigmas = SIGMAS):

	x=0

	#simage(original_image_array, "original.tif")

	images = numpy.empty((nfeatures, original_image_array.size), dtype=numpy.uint8)

	img = cv2.normalize(original_image_array,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
	images[x]=img.flatten(order='F'); x+=1

	for age in AGE_CLASSES:
		img = label_age(original_image_array, age, filename)
		#f = "AGE_" + age + ".tif";  simage(img, f)

		#img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1

	for s in sigmas:

		# Gaussian Smoothing
		img = gaussian_filter(original_image_array, sigma=s)

		#f = "gaussian_smoothing_" + str(s) + ".tif";   simage(img, f)

		img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1

		# Sobel Edge Detection
		img = scipy.ndimage.sobel(original_image_array, mode='constant', cval=s)

		#f = "sobel_edge_detection_" + str(s) + ".tif";   simage(img, f)

		img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1

		# Laplacian of Gaussian Edge Detection
		img = scipy.ndimage.gaussian_laplace(original_image_array, sigma=s)

		#f = "laplacian_of_gaussian_edge_detection_" + str(s) + ".tif";   simage(img, f)

		img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1

		# Gaussian Gradient Magnitude Edge Detection
		img = scipy.ndimage.gaussian_gradient_magnitude(original_image_array, sigma=s)

		#f = "gaussian_gradient_magnitude__edge_detection_" + str(s) + ".tif";   simage(img, f)

		img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1

		# Difference of Gaussians
		k = 1.7
		tmp1_array = scipy.ndimage.gaussian_filter(original_image_array, sigma=s*k) 
		tmp2_array = scipy.ndimage.gaussian_filter(original_image_array, sigma=s)
		img = (tmp1_array - tmp2_array)

		#f = "difference_of_gaussians_" + str(s) + ".tif";   simage(img, f)

		img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1
		
		# Structure Tensor Eigenvalues
		Axx,Axy,Ayy = feature.structure_tensor(Image.fromarray(original_image_array), sigma=s)
		large_array,small_array = feature.structure_tensor_eigvals(Axx,Axy,Ayy)

		#f = "structure_tensor_eigenvalues_large_" + str(s) + ".tif";   simage(large_array, f)
		#f = "structure_tensor_eigenvalues_small_" + str(s) + ".tif";   simage(small_array, f)

		img = cv2.normalize(large_array,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1
		img = cv2.normalize(small_array,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1

		# Hessian Matrix
		Hrr,Hrc,Hcc = feature.hessian_matrix(Image.fromarray(original_image_array), sigma=s, order='rc')

		
		#f = "hessian_matrix_Hrr_" + str(s) + ".tif";   simage(Hrr, f)
		#f = "hessian_matrix_Hrc_" + str(s) + ".tif";   simage(Hrc, f)
		#f = "hessian_matrix_Hcc_" + str(s) + ".tif";   simage(Hcc, f)


		img = cv2.normalize(Hrr,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1
		img = cv2.normalize(Hrc,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1
		img = cv2.normalize(Hcc,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
		images[x] = img.flatten(order='F'); x+=1

	# Wang points (radii of 1,3,5,7)
	newimg = wang_function(original_image_array, radius=1)

	#f = "wang_points_radius1.tif";   simage(newimg, f)

	img = cv2.normalize(newimg,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
	images[x] = img.flatten(order='F'); x+=1
	newimg = wang_function(original_image_array, radius=3)

	#f = "wang_points_radius3.tif";   simage(newimg, f)

	img = cv2.normalize(newimg,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
	images[x] = img.flatten(order='F'); x+=1
	newimg = wang_function(original_image_array, radius=5)

	#f = "wang_points_radius5.tif";   simage(newimg, f)

	img = cv2.normalize(newimg,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
	images[x] = img.flatten(order='F'); x+=1
	newimg = wang_function(original_image_array, radius=7)

	#f = "wang_points_radius7.tif";   simage(newimg, f)

	img = cv2.normalize(newimg,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
	images[x] = img.flatten(order='F'); x+=1

	# Median Neighborhood compared to overall
	newimg = median_hood(original_image_array, radius=4, factor=1.0)

	#f = "median_hood_1.0.tif";   simage(newimg, f)

	img = cv2.normalize(newimg,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
	images[x] = img.flatten(order='F'); x+=1

	newimg = median_hood(original_image_array, radius=4, factor=1.25)

	#f = "median_hood_1.25.tif";   simage(newimg, f)

	img = cv2.normalize(newimg,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
	images[x] = img.flatten(order='F'); x+=1

	newimg = median_hood(original_image_array, radius=4, factor=1.5)

	#f = "median_hood_1.5.tif";   simage(newimg, f)

	img = cv2.normalize(newimg,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
	images[x] = img.flatten(order='F'); x+=1


	# Generate proximity map (to center of image)
	newimg = center_proximity(original_image_array)

	#f = "center_proximity.tif";   simage(newimg, f)

	img = cv2.normalize(newimg,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
	images[x] = img.flatten(order='F'); x+=1

	images=numpy.transpose(images,(1,0))

	print("processed %d features out of %d" %(x,nfeatures))


	return(images)


def output_preprocessed(images, dir):
	index = 0
	for f in images:
		pathname = dir + "/" + str(index) + ".tif"
		x = Image.fromarray(f)
		x.save(pathname, "TIFF")
		index = index + 1
	
