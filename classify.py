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
import getopt
import math

import preprocess


numpy.set_printoptions(threshold=sys.maxsize)


#################################################################################
#
# HANDLE Command line arguments
#
#
def usage():
	print("python classify.pl [ --help | --verbose | --rawdir=<rawdir> | --inputlist=<inputfile> | --model=<modelfile> | --outputdir=<outputdir> ]")

try:
	opts, args = getopt.getopt(sys.argv[1:], "ho:v", ["help", "rawdir=", "inputlist=", "model=", "outputdir=" ])
except getopt.GetoptError as err:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

inputlist = None
rawdir = None
outputdir = None
model = None
verbose = False

for o, a in opts:
	if o == "-v":
		verbose = True
	elif o in ("-h", "--help"):
		print("Help, blah, blah...")
		sys.exit()
	elif o in ("-r", "--rawdir"):
		rawdir = a
	elif o in ("-i", "--inputlist"):
		inputlist = a
	elif o in ("-m", "--model"):
		model = a
	elif o in ("-o", "--outputdir"):
		outputdir = a
	else:
		assert False, "unhandled option"

if(rawdir == None or inputlist == None or outputdir == None or model == None):
	print("Missing Argument.  Exiting")
	usage()
	exit(-1)	

#
#
#################################################################################



print("RAWDIR = %s, INPUTLIST = %s, MODEL = %s, OUTPUTDIR = %s" %(rawdir, inputlist, model, outputdir))

filenames = [line.rstrip('\n') for line in open(inputlist)]


# Load the classifier model
print("Loading classifier model = %s" %model)
classifier = pickle.load(open(model,'rb'))
print("Model Loaded!")

for filename in filenames:

	print(filename)

	output = outputdir + "/" + filename

	# Load all of the files, extract all features.  This builds "dataset"
	# which is a list of feature arrays.  Every element in the list is 
	# a list of preprocessed images
	#
	rawpath = rawdir + "/" + filename
	print("Loading: %s" %rawpath)
	raw_img = Image.open(rawpath)
	img_data = preprocess.image_preprocess(f, raw_img)
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
				x = img_data[f][row][col]
				data[index][f]=x
				#print("[%d][%d]=%f" %(index,f,x))
			index = index + 1	

	# classify our input pixel
	Y_pred = classifier.predict(data)
	Y_8bit = Y_pred.astype("u1")   

	predicted_array = numpy.reshape(Y_8bit,(numrows,numcols),order='F')

	predicted_image = Image.fromarray(predicted_array)

	predicted_image.save(output, "TIFF")

	raw_img.close()
	predicted_image.close()

exit(0)

