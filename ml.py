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

MODEL_FILENAME = "./models/UNSCALED_classifier_Po1g_ALL_100trees_ALLsigmas.model"

#FILE_FILTER = "*.tif"
FILE_FILTER = "Po1g*.tif"
#FILE_FILTER = "Po1g_100_12_024*.tif"
#FILE_FILTER = "Po1g_100_12_*.tif"

THREADS = 27


# Set some paths for our image library of raw and binary labeled data
INPUT_RAW_IMAGES = "../images/raw"
INPUT_BIN_IMAGES = "../images/binary"

# The fraction of the image library that will be used for training
TRAIN_FRACTION = 0.75

# Load all of the files, extract all features.  This builds "dataset"
# which is a list of feature arrays.  Every element in the list is 
# a list of preprocessed images
nfiles = 0
dataset = []
pixels = 0
cwd = os.getcwd()
os.chdir(INPUT_RAW_IMAGES)
for f in glob.glob(FILE_FILTER):
	print("%d: [%s]" %(nfiles,f))
	nfiles=nfiles+1
	raw_img = Image.open(f)
	binpath = '../binary/' + f
	bin_img = Image.open(binpath)
	pixels = pixels + raw_img.size[0] * raw_img.size[1]
	dataset.append(preprocess.image_preprocess(raw_img, bin_img))
	raw_img.close()
	bin_img.close()
os.chdir(cwd)

print("HERE")
preprocess.output_preprocessed(dataset[0], "debug")



num_features = preprocess.feature_count()
print("Total Pixels: %d" %pixels)
print("Feature Vector Length: %d" %num_features)



#
# Now we need to reorganize the data into a long table of one pixel on each
# row, with columns representing the features
#
print("\nRe-structuring loaded data for training...\n")

fcount = 0 
index = 0

# declare our new numpy array 
data = numpy.ndarray(shape=(pixels,num_features), dtype=numpy.float32)
for file in dataset:
	numrows=len(file[0])
	numcols=len(file[0][0])

	for col in range(numcols):
		for row in range(numrows):
			for f in range(num_features):
				data[index][f]=file[f][row][col]
			index = index + 1

	# The code below prints nice status message
	j = (fcount+1)/nfiles
	sys.stdout.write('\r')
	sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
	sys.stdout.flush()

	fcount = fcount + 1

print("\n")
print("pixels = %d, INDEX = %d" %(pixels,index))

# Free up space in dataset
del dataset

# Divide dataset into attributes (X) and labels (Y)
X = data[:,1:num_features]
Y = data[:,0]

# free up space from data structure
del data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(1-TRAIN_FRACTION), random_state=0)

# print out testing and traininig sizes
print("X_train: %d, X_test: %d, Y_train: %d, Y_test: %d" %(len(X_train), len(X_test), len(Y_train), len(Y_test)))

# Scale features to a common scale (maybe not needed?)
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Train the model
print("Training the Random Forest")
classifier = RandomForestClassifier(n_estimators=100, max_depth=32, verbose=2, n_jobs=THREADS, random_state=0)
classifier.fit(X_train, Y_train)
print("DUMPING MODEL")
pickle.dump(classifier, open(MODEL_FILENAME,'wb'))

# generate predictions against our test set
Y_pred = classifier.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

exit(0)


