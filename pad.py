from PIL import Image, ImageFilter, ImageOps
import os
import getopt
import yaml
import pickle
from os import listdir
from os.path import isfile, join
import numpy
import scipy
import cv2
import pprint
from scipy.ndimage.filters import gaussian_filter
from skimage import feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import glob
import sys
from time import sleep, time
from datetime import datetime
from random import seed
from random import random

import preprocess



# set our random seed based on current time
now = int(time())

# SHENEMAN - UNCOMMENT SEED CALL BELOW WHEN DONE TESTING
seed(now)
numpy.random.seed(now)


#################################################################################
#
# HANDLE Command line arguments
#
#
def usage():
	print("python pad.py [ --help | --verbose | --config=<YAML config filename> ] ")

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
	configfile="pad.yaml"
	#print("Missing Argument.  Exiting")
	#usage()
	#exit(-1)

#################################################################################

#################################################################################
#
# Format and Example config YAML file:
#
# FORMAT:
# -------
#   rawdir: <path to input raw images>
#   outdir: <path to output directory for padded images>
#
# EXAMPLE:
# --------
#   rawdir:             "../images/raw"
#   outdir:             "../images/padded"
#
#################################################################################

cf = open(configfile, "r")
config = yaml.load(cf, Loader=yaml.FullLoader)
print("YAML CONFIG:")
for c in config:
	print("    [%s]:\"%s\"" %(c,config[c]))
print("\n")
cf.close()



#################################################################################################
##  FUNCTION DEFINITIONS
#################################################################################################


##################################################################################################
#
# function:  pad_file()
#

def pad_file(filename,new_width,new_height):
	
	print("In pad_file():  file = %s" %filename)
	fullpath = config["rawdir"] + '/' + filename
	img = Image.open(fullpath)
	(width,height) = img.size

	delta_w = new_width  - width
	delta_h = new_height - height

	padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
	new_img = ImageOps.expand(img, padding)	

	return(new_img)
	



##################################################################################################
#
# function:  unpad_file()
#
def unpad_image(image, filename):
	
	print("In unpad_image():  file = %s" %filename)

	(width,height) = image.size	

	# determine original image width and height
	fullpath = config["rawdir"] + '/' + filename
	img = Image.open(fullpath)
	(original_width,original_height) = img.size

	delta_w = width  - original_width
	delta_h = height - original_height

	border = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
	new_img = ImageOps.crop(image, border)

	return(new_img)

	



##################################################################################################
#
# function:  get_maxima()
#
def get_maxima(filenames):

	# allocate an array for images
	num_images = len(filenames)

	# Read raw and binary images into these preallocated numpy arrays of objects
	max_width = 0;
	max_height = 0;
	for f in filenames:
		rawpath = config["rawdir"] + "/" + f
	
		print(rawpath)

		raw_image = Image.open(rawpath)
		(width,height) = raw_image.size;
		if(width > max_width):
			max_width = width

		if(height > max_height):
			max_height = height

	return(width,height)








filenames = listdir(config["rawdir"]) 

for filename in filenames:
	padded_image = pad_file(filename,128,128)
	
	savepath = config["outdir"] + '/' + filename
	padded_image.save(savepath, "TIFF")

exit(0)



#width,height) = get_maxima(filenames)
#print("Maximum Width: %d" %width)
#print("Maximum Height: %d" %height)

