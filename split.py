import os
from os import listdir
from os.path import isfile, join
import pprint
import glob
import sys
from time import time
from random import seed, random, shuffle
from math import floor


# set our random seed based on current time
now = int(time())
seed(now)

# Set some paths for our image library of raw and binary labeled data
IMG_RAWPATH = "../images/raw"
IMG_BINPATH = "../images/binary"

# Set output filenames
TRAINSET_FILENAME = "train.txt"
TESTSET_FILENAME = "test.txt"

FILE_FILTER = "*.tif"

# The fraction of the image library that will be used for training
TRAIN_FRACTION = 0.75

# Get all of the filenames that match the filter and shuffle them in place
cwd = os.getcwd()
os.chdir(IMG_RAWPATH)
filenames = glob.glob(FILE_FILTER)
os.chdir(cwd)
shuffle(filenames)

num_filenames = len(filenames)

if(TRAIN_FRACTION < 0 or TRAIN_FRACTION > 1.0):
	print("ERROR: TRAIN_FRACTION must be between 0,0 and 1,0")
	exit(-1)

train_partition_start = 0
train_partition_end = floor(num_filenames*TRAIN_FRACTION-1)
if(train_partition_end == num_filenames-1):
	test_partition_start = -1
	test_partition_end   = -1
else:
	test_partition_start = train_partition_end+1
	test_partition_end   = num_filenames-1

if(train_partition_start < 0 or train_partition_end < 0 or train_partition_end < train_partition_start):
	train_partition_start = -1
	train_partiiton_end = -1
	

print("TOTAL NUMBER OF FILENAMES: %d" %num_filenames)
print("TRAIN: %d thru %d" %(train_partition_start,train_partition_end))
print("TEST: %d thru %d" %(test_partition_start,test_partition_end))

# Write the training set input file
if(train_partition_start >= 0):
	train_file = open(TRAINSET_FILENAME, "w")
	for i in range(train_partition_start, train_partition_end+1):
		train_file.write(filenames[i] + "\n")
	train_file.close()

# Write the test set input file
if(test_partition_start >= 0):
	test_file = open(TESTSET_FILENAME, "w")
	for i in range(test_partition_start, test_partition_end+1):
		test_file.write(filenames[i] + "\n")
	test_file.close()

exit(0)


