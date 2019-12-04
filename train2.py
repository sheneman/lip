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

THREADS = 30

# Set some paths for our image library of raw and binary labeled data
IMG_RAWPATH = "../images/raw"
IMG_BINPATH = "../images/binary"

# Set the training and testing filenames here
TRAINSET_FILENAME = "./train.txt"
TESTSET_FILENAME  = "./test.txt"

trainset_filenames = [line.rstrip('\n') for line in open(TRAINSET_FILENAME)]
testset_filenames = [line.rstrip('\n') for line in open(TESTSET_FILENAME)]

# Load all of the files from the training set, extract all features.  This builds "dataset"
# which is a list of feature arrays.  Every element in the list is 
# a list of preprocessed images
nfiles = 0
raw_training_dataset = []
bin_training_dataset = []
pixels = 0
for f in trainset_filenames:
	print("%d: [%s]" %(nfiles,f))
	nfiles=nfiles+1
	rawpath = IMG_RAWPATH + "/" + f
	binpath = IMG_BINPATH + "/" + f
	raw_img = Image.open(rawpath)
	bin_img = Image.open(binpath)
	pixels = pixels + raw_img.size[0] * raw_img.size[1]
	raw_training_dataset.append(preprocess.image_preprocess(f, raw_img))
	bin_training_dataset.append(numpy.array(bin_img))
	raw_img.close()
	bin_img.close()

# this spews out all of the preprocessed images for the first image into a folder
preprocess.output_preprocessed(raw_training_dataset[0], "debug")

num_features = preprocess.feature_count()
print("Total Pixels: %d" %pixels)
print("Feature Vector Length: %d" %num_features)

exit(0)


#
# Now we need to reorganize the data into a long table of one pixel on each
# row, with columns representing the features
#
print("\nRe-structuring loaded data for training...\n")

fcount = 0 
index = 0

# declare our new numpy array for handling raw data
raw_data = numpy.ndarray(shape=(pixels,num_features), dtype=numpy.float32)
for file in raw_dataset:
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

# separately handle the binary labels
index = 0
bin_data = numpy.ndarray(shape=pixels, dtype=numpy.float32)
for file in bin_dataset:
	numrows=len(file)
	numcols=len(file[0])

	for col in range(numcols):
		for row in range(numrows):
			bin_data[index]=file[row][col]
			index = index + 1

print("\n")
print("pixels = %d, INDEX = %d" %(pixels,index))

# Free up space in dataset
del raw_dataset
del bin_dataset

X_train, X_test, Y_train, Y_test = train_test_split(raw_data, bin_data, test_size=(1-TRAIN_FRACTION))
# free up space from data structure
del raw_data
del bin_data

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


