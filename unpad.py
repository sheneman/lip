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
	print("python unpad.py [ --help | --verbose | --config=<YAML config filename> ] ")

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
	configfile="unpad.yaml"
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
#   rawdir: <path to original raw images>
#   padded: <path to padded images to unpad>
# unpadded: <path to place the unpadded images>
#
# EXAMPLE:
# --------
#   rawdir:             "../images/raw"
#   padded:             "./output"
# unpadded:             "./unpadded"
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

	

filenames = listdir(config["padded"]) 

for f in filenames:
	fullpath = config["padded"] + '/' + f
	img = Image.open(fullpath)
	unpadded_image = unpad_image(img, f)
	
	savepath = config["unpadded"] + '/' + f
	unpadded_image.save(savepath, "TIFF")

exit(0)

