from PIL import Image, ImageFilter
import os
import getopt
import yaml
import pickle
from os import listdir
from os.path import isfile, join
import numpy
import scipy
import pprint
from scipy.ndimage.filters import gaussian_filter
from skimage import feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
seed(now)


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
#   mask: <0 or 1> - whether to use masks or not
#   rawdir: <path to raw images>
#   bindir: <path to bin images>
#   maskdir: <path to binary image mask files>
#   trainlist: <input path for training images>
#   validationlist: <input path for validation images>
#   testlist: <input path for test images>
#
# EXAMPLE:
# --------
#   modelname:		"./models/new_model_file.model"
#   threads:		25
#   mask:		1
#   rawdir:             "../images/raw"
#   bindir:             "../images/binary"
#   maskdir:		"../images/mask"
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



##################################################################################################
#
# function:  build_dataset()
#
# Load all of the images from the specified files, extract all features.  This builds "dataset"
# which is a list of feature arrays.  Every element in the list is 
# a list of preprocessed images

def build_dataset(filenames):
	nfiles = 0
	raw = []
	bin = []
	pixel_cnt = 0
	for f in filenames:
		print("%d: [%s]" %(nfiles,f))
		nfiles=nfiles+1
		rawpath = config["rawdir"] + "/" + f
		binpath = config["bindir"] + "/" + f
		raw_img = Image.open(rawpath)
		bin_img = Image.open(binpath)

		if(config["mask"] == 1):
			maskpath = config["maskdir"] + "/" + f
			mask_img = Image.open(maskpath)
			raw_img = apply_mask(raw_img, mask_img)

		pixel_cnt = pixel_cnt + raw_img.size[0] * raw_img.size[1]
		raw.append(preprocess.image_preprocess(f, raw_img))
		bin.append(numpy.array(bin_img))
		raw_img.close()
		bin_img.close()
	return(raw, bin, pixel_cnt)




##################################################################################################
#
# function: restructure_data()
#
# Now we need to reorganize the data into a long table of one pixel on each
# row, with columns representing the features
#
def restructure_data(raw, binary, pixel_cnt, num_features):

	nfiles = len(raw)

	fcount = 0 
	index = 0

	# declare our new numpy array for handling raw data
	raw_data = numpy.ndarray(shape=(pixel_cnt,num_features), dtype=numpy.float32)
	print("Restructuring raw data...")
	for file in raw:
		numrows=len(file[0])
		numcols=len(file[0][0])

		for col in range(numcols):
			for row in range(numrows):
				for f in range(num_features):
					raw_data[index][f]=file[f][row][col]
				index = index + 1

		# The code below prints nice status message
		j = (fcount+1)/nfiles
		sys.stdout.write('\r')
		sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
		sys.stdout.flush()

		fcount = fcount + 1
	print("\n")

	# separately handle the binary labels
	fcount = 0
	index = 0
	bin_data = numpy.ndarray(shape=pixel_cnt, dtype=numpy.float32)
	print("Restructuring binary data...")
	for file in binary:
		numrows=len(file)
		numcols=len(file[0])

		for col in range(numcols):
			for row in range(numrows):
				bin_data[index]=file[row][col]
				index = index + 1

		# The code below prints nice status message
		j = (fcount+1)/nfiles
		sys.stdout.write('\r')
		sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
		sys.stdout.flush()

		fcount = fcount + 1

	print("\n")

	return(raw_data, bin_data)

#################################################################################################


#
#  MAIN CODE HERE
#


train_filenames      = [line.rstrip('\n') for line in open(config["trainlist"])]
validation_filenames = [line.rstrip('\n') for line in open(config["validationlist"])]
test_filenames       = [line.rstrip('\n') for line in open(config["testlist"])]

print("Loading Training Image Data...")
(train_raw, train_bin, train_pixel_cnt)                = build_dataset(train_filenames)
print("Loading Validation Image Data...")
(validation_raw, validation_bin, validation_pixel_cnt) = build_dataset(validation_filenames)
print("Loading Testing Image Data...")
(test_raw, test_bin, test_pixel_cnt)                   = build_dataset(test_filenames)


# this spews out all of the preprocessed images for the first image into a folder
preprocess.output_preprocessed(train_raw[0], "debug")

num_features = preprocess.feature_count()
print("Train Pixels: %d" %train_pixel_cnt)
print("Validation Pixels: %d" %validation_pixel_cnt)
print("Test Pixels: %d" %test_pixel_cnt)
print("Feature Vector Length: %d" %num_features)


print("Serializing Training data:")
(X_train, Y_train)           = restructure_data(train_raw, train_bin, train_pixel_cnt, num_features)
print("Serializing Validation data:")
(X_validation, Y_validation) = restructure_data(validation_raw, validation_bin, validation_pixel_cnt, num_features)
print("Serializing Test data:")
(X_test, Y_test)             = restructure_data(test_raw, test_bin, test_pixel_cnt, num_features)

#X_train, X_test, Y_train, Y_test = train_test_split(raw_data, bin_data, test_size=(1-TRAIN_FRACTION))
# free up space from data structure
#del raw_data
#del bin_data

# print out testing and traininig sizes
print("X_train: %d, X_test: %d, Y_train: %d, Y_test: %d" %(len(X_train), len(X_test), len(Y_train), len(Y_test)))

# Train the model
print("Training the Random Forest")
#classifier = RandomForestClassifier(n_estimators=100, max_depth=32, verbose=2, n_jobs=config["threads"])
classifier = RandomForestClassifier(n_estimators=100, verbose=2, n_jobs=config["threads"])
classifier.fit(X_train, Y_train)
print("DUMPING MODEL")
pickle.dump(classifier, open(config["modelname"],'wb'))

# generate predictions against our test set
Y_pred = classifier.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

exit(0)


