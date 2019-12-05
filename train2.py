from PIL import Image, ImageFilter
import os
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



MODEL_FILENAME = "./models/classifier_MTYL17_100trees_no-maxdepth-ALLsigmas.model"

THREADS = 7

# Set some paths for our image library of raw and binary labeled data
IMG_RAWPATH = "../images/raw"
IMG_BINPATH = "../images/binary"

# Set the training and testing filenames here
TRAIN_FILENAME      = "./train_list.txt"
VALIDATION_FILENAME = "./validation_list.txt"
TEST_FILENAME       = "./test_list.txt"



#################################################################################################
##  FUNCTION DEFINITIONS
#################################################################################################
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
		rawpath = IMG_RAWPATH + "/" + f
		binpath = IMG_BINPATH + "/" + f
		raw_img = Image.open(rawpath)
		bin_img = Image.open(binpath)
		pixel_cnt = pixel_cnt + raw_img.size[0] * raw_img.size[1]
		raw.append(preprocess.image_preprocess(f, raw_img))
		bin.append(numpy.array(bin_img))
		raw_img.close()
		bin_img.close()
	return(raw, bin, pixel_cnt)


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


train_filenames      = [line.rstrip('\n') for line in open(TRAIN_FILENAME)]
validation_filenames = [line.rstrip('\n') for line in open(VALIDATION_FILENAME)]
test_filenames       = [line.rstrip('\n') for line in open(TEST_FILENAME)]

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
#classifier = RandomForestClassifier(n_estimators=100, max_depth=32, verbose=2, n_jobs=THREADS)
classifier = RandomForestClassifier(n_estimators=100, verbose=2, n_jobs=THREADS)
classifier.fit(X_train, Y_train)
print("DUMPING MODEL")
pickle.dump(classifier, open(MODEL_FILENAME,'wb'))

# generate predictions against our test set
Y_pred = classifier.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

exit(0)


