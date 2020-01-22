from PIL import Image, ImageFilter
import os 
import getopt
import yaml
import pickle
from os import listdir
from os.path import isfile, join
import numpy
import scipy
import cv2
import pandas as pd
import pprint
from scipy.ndimage.filters import gaussian_filter
from skimage import feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import glob
from time import sleep
import sys
import math

import preprocess


numpy.set_printoptions(threshold=sys.maxsize)


#################################################################################
#
# HANDLE Command line arguments
#
#
def usage():
	print("python train2.pl [ --help | --verbose | --config=<YAML config filename> ] ")

try:
	opts, args = getopt.getopt(sys.argv[1:], "ho:v", ["help", "config="])
except getopt.GetoptError as err:
	print(err)  # will print something like "option -a not recognized"
	usage()
	sys.exit(2)

configfile = None
verbose = False

for o, a in opts:
	if o == "-v":
		verbose = True
	elif o in ("-h", "--help"):
		print("Help, blah, blah...")
		sys.exit()
	elif o in ("-c", "--config"):
		configfile = a
	else:
		assert False, "unhandled option"

if(configfile == None):
	print("Missing Argument.  Exiting")
	usage()
	exit(-1)

#################################################################################

#################################################################################
#
# Format and Example config YAML file:
#
# FORMAT:
# -------
#   inputlist:	<path to a text file that names which files to classify>
#   rawdir: 	<path to raw images>
#   model:	<path to the model to use for classification>
#   outputdir:	<path to output folder>
#   mask: 	<0 or 1> - whether to use masks or not
#   maskdir: 	<path to binary image mask files> 
#
# EXAMPLE:
# --------
#   inputlist:	"./input_filenames.txt"
#   rawdir:	"../images/raw"
#   model:	"./models/foo.model"
#   outputdir:	"./output_directory"
#   mask:	1
#   maskdir:	"../images/mask"
#
#################################################################################

cf = open(configfile, "r")
config = yaml.load(cf, Loader=yaml.FullLoader)
print("YAML CONFIG:")
for c in config:
	print("    [%s]:\"%s\"" %(c,config[c]))
print("\n")
cf.close()

# get the list of raw files to classify
filenames = [line.rstrip('\n') for line in open(config["inputlist"])]

# Load the classifier model
print("Loading classifier model = %s" %config["model"])
classifier = pickle.load(open(config["model"],'rb'))
print("Model Loaded!")

for filename in filenames:

	print(filename)

	output = config["outputdir"] + "/" + filename

	# Load all of the files, extract all features.  This builds "dataset"
	# which is a list of feature arrays.  Every element in the list is 
	# a list of preprocessed images
	#
	rawpath = config["rawdir"] + "/" + filename
	print("Loading: %s" %rawpath)
	raw_img = Image.open(rawpath)

	raw_cv2 = numpy.array(raw_img)

	raw_cv2 = cv2.normalize(raw_cv2,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
	raw_cv2 = cv2.equalizeHist(raw_cv2)

	raw_img.close()
	raw_img = Image.fromarray(raw_cv2)

	img_data = preprocess.image_preprocess(filename, raw_img)
	numcols,numrows = raw_img.size

	num_features = preprocess.feature_count()
	print("Image Size: numcols=%d x numrows=%d" %(numcols,numrows))
	print("Num Features: %d" %num_features)

	#
	# Now we need to reorganize the data into a long table of one pixel on each
	# row, with columns representing the features
	#
	print("\nRe-structuring loaded data for classification...\n")

	fcount = 0 
	index = 0

	# declare our new numpy array 
	data = numpy.ndarray(shape=((numrows*numcols),num_features), dtype=numpy.uint8)

	for col in range(numcols):
		for row in range(numrows):
			for f in range(num_features):
				x = img_data[f][row][col]
				data[index][f]=x
				#print("[%d][%d]=%f" %(index,f,x))
			index = index + 1	

	# classify our input pixel
	Y_pred = classifier.predict(data)
	#Y_8bit = Y_pred.astype("u1")   

	#predicted_array = numpy.reshape(Y_8bit,(numrows,numcols),order='F')
	predicted_array = numpy.reshape(Y_pred,(numrows,numcols),order='F')

	predicted_image = Image.fromarray(predicted_array)
	
	# Handles masks, if specified in config
	if(config["mask"] != 0):
		maskpath = config["maskdir"] + "/" + filename
		mask_img = Image.open(maskpath)
		predicted_image = preprocess.apply_mask(predicted_image, mask_img)
	
	predicted_image.save(output, "TIFF")

	raw_img.close()
	predicted_image.close()

exit(0)

