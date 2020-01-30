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
import pprint
from scipy.ndimage.filters import gaussian_filter
from skimage import feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

import preprocessnew as preprocess



# set our random seed based on current time
now = int(time())

# SHENEMAN - UNCOMMENT SEED CALL BELOW WHEN DONE TESTING
seed(2)
numpy.random.seed(2)


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
#   modelname: path and filename for pickled model output
#   threads: integer specifying the number of parallel threads to use
#   rawdir: <path to raw images>
#   bindir: <path to bin images>
#   feat_select: <0 or 1> - whether to do explicit feature selection or not
#   feat_thresh: <String: "mean" or "median" OR float representing threshold for keeping feature
#   feat_max: <Integer or  None"> - The maximum number of features above feat_thresh to retain
#   trainlist: <input path for training images>
#   validationlist: <input path for validation images>
#   testlist: <input path for test images>
#
# EXAMPLE:
# --------
#   modelname:		"./models/new_model_file.model"
#   threads:		25
#   rawdir:             "../images/raw"
#   bindir:             "../images/binary"
#   feat_select:        1
#   feat_thresh:        "mean"
#   feat_max:           "None"
#   trainlist:		"./trainlist.txt"
#   validationlist:  	"./validationlist.txt"
#   testlist:        	"./testlist.txt"
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
# function:  build_dataset()
#
# Load all of the images from the specified files, extract all features.  This builds "dataset"
# which is a list of feature arrays.  Every element in the list is 
# a list of preprocessed images

def build_dataset(filenames, nfeatures):

	# allocate an array for images
	num_images = len(filenames)
	raw_images = numpy.empty(num_images, dtype=object)
	bin_images = numpy.empty(num_images, dtype=object)

	# Read raw and binary images into these preallocated numpy arrays of objects
	index = 0
	pixel_cnt = 0
	for f in filenames:
		print("%d: [%s]" %(index,f))
		rawpath = config["rawdir"] + "/" + f
		binpath = config["bindir"] + "/" + f

		raw_image = Image.open(rawpath)
		raw_images[index] = numpy.array(raw_image)
		raw_image.close()

		bin_image = Image.open(binpath)
		bin_images[index] = numpy.array(bin_image)
		bin_image.close()
	
		pixel_cnt += raw_images[index].size
		index += 1

	print("Number of Pixels in %d images: %d" %(index,pixel_cnt))


	# Now that we know the number of pixels, we can allocate raw and bin arrays
	raw = numpy.empty((pixel_cnt,nfeatures),dtype=numpy.uint8)
	bin = numpy.empty(pixel_cnt,dtype=numpy.uint8)


	#
	# Process raw images
	#
	pixel_cnt = 0
	pixel_index = 0
	for raw_cv2 in raw_images:

		pixels = preprocess.image_preprocess(f, raw_cv2)
		raw[pixel_index:pixel_index+pixels.shape[0],:] = pixels

		pixel_index+=pixels.shape[0]
		pixel_cnt += raw_cv2.size


	#
	# Process binary images
	#
	pixel_index = 0
	for bin_cv2 in bin_images:

		pixels = bin_cv2.flatten(order='F')
		bin[pixel_index:pixel_index+len(pixels)] = pixels

		pixel_index += len(pixels)
	

	return(raw, bin, pixel_cnt)


#################################################################################################


#
#  MAIN CODE HERE
#


#
# get feature labels
#
flabels   = preprocess.feature_labels()
nfeatures = preprocess.feature_count()
print("Number of Feature Labels: %d" %(len(flabels)))
print(flabels)


train_filenames      = [line.rstrip('\n') for line in open(config["trainlist"])]
validation_filenames = [line.rstrip('\n') for line in open(config["validationlist"])]
test_filenames       = [line.rstrip('\n') for line in open(config["testlist"])]

print("Loading Training Image Data...")
(X_train, Y_train, train_pixel_cnt)                = build_dataset(train_filenames, nfeatures)
print("Loading Validation Image Data...")
(X_validation, Y_validation, validation_pixel_cnt) = build_dataset(validation_filenames, nfeatures)
print("Loading Testing Image Data...")
(X_test, Y_test, test_pixel_cnt)                   = build_dataset(test_filenames, nfeatures)

#print(type(X_train))
#print(X_train.shape)
#print(X_train.dtype)
#print(type(Y_train))
#print(Y_train.shape)
#print(Y_train.dtype)
#print(type(X_validation))
#print(X_validation.shape)
#print(type(Y_validation))
#print(Y_validation.shape)
#print(type(X_test))
#print(X_test.shape)
#print(X_test.dtype)
#print(type(Y_test))
#print(Y_test.shape)
#print(Y_test.dtype)
#
#print(X_train[1])
#print(X_train[len(X_train)-1])
#print(X_train[3])
#print(X_train[len(X_train)-3])
#print(X_test[0])
#print(X_test[len(X_test)-1])



# this spews out all of the preprocessed images for the first image into a folder
#preprocess.output_preprocessed(train_raw[0], "debug")

print("Train Pixels: %d" %train_pixel_cnt)
print("Validation Pixels: %d" %validation_pixel_cnt)
print("Test Pixels: %d" %test_pixel_cnt)
print("Feature Vector Length: %d" %nfeatures)


# print out testing and training sizes
print("X_train: %d, X_test: %d, Y_train: %d, Y_test: %d" %(len(X_train), len(X_test), len(Y_train), len(Y_test)))

# Train the model
print("Training the Random Forest")
#classifier = RandomForestClassifier(n_estimators=100, max_depth=32, verbose=2, n_jobs=config["threads"])
classifier = RandomForestClassifier(n_estimators=100, verbose=2, n_jobs=config["threads"])

classifier.fit(X_train, Y_train)

# Output the most important features to a telemetry file

feature_file = open(config["importance"], "w")
for feature in sorted(zip(flabels, classifier.feature_importances_), key=lambda x: x[1], reverse=True):
	feature_file.write("%s,%f\n" %feature)
feature_file.close()


print("DUMPING MODEL")
pickle.dump(classifier, open(config["modelname"],'wb'))

# generate predictions against our test set
Y_pred = classifier.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

exit(0)


