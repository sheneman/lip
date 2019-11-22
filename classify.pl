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
from time import sleep
import sys

import preprocess



numpy.set_printoptions(threshold=sys.maxsize)


MODEL_FILENAME = "./models/classifier_Po1g_ALL_100trees.model"

filename = "MTYL_100_10_001_SLIM_A.tif"


# Set some paths for our image library of raw and binary labeled data
INPUT_RAW_IMAGES = "../images/raw"


# Load the classifier model
print("Loading classifier model = %s" %MODEL_FILENAME)
classifier = pickle.load(open(MODEL_FILENAME,'rb'))

# Load all of the files, extract all features.  This builds "dataset"
# which is a list of feature arrays.  Every element in the list is 
# a list of preprocessed images
#
rawpath = "../images/raw/" + filename
binpath = '../images/binary/' + filename
print("Loading: %s" %rawpath)
raw_img = Image.open(rawpath)
bin_img = Image.open(binpath)
img_data = preprocess.image_preprocess(raw_img, bin_img)
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
data = numpy.ndarray(shape=((numrows*numcols),num_features), dtype=numpy.float32)

for col in range(numcols):
	for row in range(numrows):
		for f in range(num_features):
			data[index][f]=img_data[f][row][col]
		index = index + 1	


# generate predictions against our test set
# Divide dataset into attributes (X) and labels (Y)
X_data = data[:,1:num_features]
Y_data = data[:,0]

Y_pred = classifier.predict(X_data)
#print(Y_pred)
#print(Y_data)

predicted_array = numpy.reshape(Y_pred,(numrows,numcols),order='F')
labels_array = numpy.reshape(Y_data,(numrows,numcols),order='F')


predicted_image = Image.fromarray(predicted_array)
labels_binary  = Image.fromarray(labels_array)

predicted_image.save("output/predicted.tif", "TIFF")
labels_binary.save("output/labels.tif", "TIFF")
bin_img.save("output/original_binary.tif", "TIFF")

raw_img.close()
bin_img.close()

exit(0)


