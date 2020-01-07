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
#   inputlist:	<path to a text file that names which files to classify>
#   rawdir: 	<path to raw images>
#   model:	<path to the model to use for classification>
#   outputdir:	<path to output folder>
#
# EXAMPLE:
# --------
#   inputlist:	"./input_filenames.txt"
#   rawdir:	"../images/raw"
#   model:	"./models/foo.model"
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


print("RAWDIR = %s, INPUTLIST = %s, MODEL = %s, OUTPUTDIR = %s" %(config["rawdir"], config["inputlist"], config["model"], config["outputdir"]))

filenames = [line.rstrip('\n') for line in open(config["inputlist"])]


# Load the classifier model
print("Loading classifier model = %s" %config["model"])
classifier = pickle.load(open(config["model"],'rb'))
print("Model Loaded!")

for filename in filenames:

	print(filename)

	output = config["outputdir"] + "/" + filename

	# Load all of the files, extract all features.  This builds "dataset"
	# which is a list of feature arrays.  Every element in the list is 
	# a list of preprocessed images
	#
	rawpath = config["rawdir"] + "/" + filename
	print("Loading: %s" %rawpath)
	raw_img = Image.open(rawpath)
	img_data = preprocess.image_preprocess(filename, raw_img)
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

