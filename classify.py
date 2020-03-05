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
from xgboost import XGBClassifier
from skimage import feature
from sklearn.neural_network import MLPClassifier
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
	print("python classify.py [ --help | --verbose | --config=<YAML config filename> ] ")

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
	configfile="classify.yaml"
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
#   inputlist:	<path to a text file that names which files to classify>
#   rawdir: 	<path to raw images>
#   rf_model:	<path to the Random Forest model to use for classification>
#   xgb_model:  <path to the XGBoost model to use for classification> 
#   mlp_model:  <path to the Multi-Level Perceptron model to use for classification>
#   outputdir:	<path to output folder>
#
# EXAMPLE:
# --------
#   inputlist:	"./input_filenames.txt"
#   rawdir:	"../images/raw"
#   rf_model:	"./models/rf.model"
#   xgb_model:	"./models/xgb.model"
#   mlp_model:	"./models/mlp.model"
#   outputdir:	"./output_directory"
#
#################################################################################

cf = open(configfile, "r")
config = yaml.load(cf, Loader=yaml.FullLoader)
print("YAML CONFIG:")
for c in config:
	print("    [%s]:\"%s\"" %(c,config[c]))
print("\n")
cf.close()

#
# get feature labels
#
flabels   = preprocess.feature_labels()
nfeatures = preprocess.feature_count(flabels)
print("Number of Feature Labels: %d" %nfeatures)


# get the list of raw files to classify
filenames = [line.rstrip('\n') for line in open(config["inputlist"])]



# Load the XGB Classifier
print("Loading XGBoost Classifier = %s" %config["xgb_model"])
xgb_classifier = pickle.load(open(config["xgb_model"],'rb'))
xgb_classifier.verbose = False
print("XGBoost Classifier Loaded!")

# Load the Random Forest Classifier
print("Loading Random Forest Classifier = %s" %config["rf_model"])
rf_classifier = pickle.load(open(config["rf_model"],'rb'))
rf_classifier.verbose = False
print("Random Forest (RF) Classifier Loaded!")

# Load the Neural Network (MLP) Classifier
print("Loading Neural Network Classifier = %s" %config["mlp_model"])
mlp_classifier = pickle.load(open(config["mlp_model"],'rb'))
mlp_classifier.verbose = False
print("Neural Network (MLP) Classifier Loaded!")



for filename in filenames:

	print(filename)

	output = config["outputdir"] + "/" + filename

	######################################################################
	# Load all of the files, extract all features.  This builds "dataset"
	# which is a list of feature arrays.  Every element in the list is 
	# a list of preprocessed images
	#
	rawpath = config["rawdir"] + "/" + filename
	print("Loading: %s" %rawpath)
	raw_img = Image.open(rawpath)
	numcols,numrows = raw_img.size
	raw_cv2 = numpy.array(raw_img)
	raw_img.close()

	data = preprocess.image_preprocess(filename, nfeatures, raw_cv2)

	print("Image Size: numcols=%d x numrows=%d" %(numcols,numrows))
	print("Num Features: %d" %nfeatures)

	# classify our input pixels 
	RF_Y_pred  =  rf_classifier.predict(data)    # Random Forest
	MLP_Y_pred = mlp_classifier.predict(data)   # Neural Network
	XGB_Y_pred = xgb_classifier.predict(data)   # XGBoost Classifier


	# Majority votes for lipid pixels across ML methods
	pred = ((RF_Y_pred.astype(numpy.uint16) + MLP_Y_pred.astype(numpy.uint16) + XGB_Y_pred.astype(numpy.uint16))/255).astype(numpy.uint8)
	
	print("Lipid Pixels with only ONE vote: %d" %(numpy.sum(pred == 1)))
	print("Lipid Pixels with TWO votes: %d" %(numpy.sum(pred == 2)))
	print("Lipid Pixels with THREE votes: %d" %(numpy.sum(pred == 3)))

	pred[pred < 2] = 0
	pred[pred >=2] = 255

	predicted_array = numpy.reshape(pred,(numrows,numcols),order='F')
	predicted_array = cv2.normalize(predicted_array,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
	predicted_image = Image.fromarray(predicted_array)
	
	predicted_image.save(output, "TIFF")


	predicted_image.close()

exit(0)

