from PIL import Image, ImageFilter
import numpy
import scipy
import imageio
import cv2
import pprint
from scipy.ndimage.filters import gaussian_filter
from skimage import feature

# our functions
import preprocess


filenames	= [line.rstrip('\n') for line in open("trainlist.txt")]
RAW_DIR 	= "../images/raw"
OUT_DIR		= "./circlesout"



#INPUTFILE = "../images/raw/MTYL_100_1_003_SLIM_B.tif"
#INPUTFILE = "../images/raw/MTYL_100_14_003_SLIM_G.tif"


def sample_intensity_within_circle(img, center, radius):
	mask = numpy.zeros(img.shape, numpy.uint8)
	
	cv2.circle(mask, center, radius, 255, -1)	
	raw_img = Image.fromarray(mask)
	raw_img.save("mask.tif")
	
	where = numpy.where(mask==255)
	print("img.shape = (%d,%d)" %(img.shape))
	print("WHERE:")
	print(where)
	print("DOOM")
	intensities = img[where[0],where[1]]
	pprint.pprint(intensities,depth=10000)
	return(numpy.mean(intensities))
	



for f in filenames:

	print(f)
	fullpath = RAW_DIR + '/' + f	
	raw_img = Image.open(fullpath)
	raw_cv2 = numpy.array(raw_img)

	img8 = cv2.normalize(raw_cv2,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
	img8 = cv2.equalizeHist(img8)

	#ret,img8 = cv2.threshold(img8,200,255,cv2.THRESH_BINARY)

	img_canny = preprocess.auto_canny(img8,sigma=0.33)
	x = Image.fromarray(img_canny)
	p = "canny" + '/' + f
	x.save(p)
	

	circles = cv2.HoughCircles(image=img8, method=cv2.HOUGH_GRADIENT, dp=1.2, param1=5, param2=36,minDist=25,minRadius=3,maxRadius=20)
	if circles is not None:

		circles = numpy.round(circles[0, :]).astype("int")
		output = cv2.cvtColor(img8.copy(),cv2.COLOR_GRAY2BGR)
		remove_circles = []
	 
		# loop over the (x, y) coordinates and radius of the circles
		index = 0
		for (x, y, r) in circles:
			# draw the circle in the output image
			#if(sample_intensity_within_circle(img8_thresh,(y,x),r) > 127):
			cv2.circle(output, (x, y), r, (0, 255, 0), 1)
			#else:
			#	remove_circles.append(index)

			index = index + 1

		circles = numpy.delete(circles,remove_circles,0)

		raw_img = Image.fromarray(output)
		outpath = OUT_DIR + '/' + f
		raw_img.save(outpath)
		pprint.pprint(circles)
	else:
		print("No Circles Identified")

