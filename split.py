import os
from os import listdir
from os.path import isfile, join
import pprint
import glob
import sys
from time import time
from random import seed, random, shuffle
from math import floor

#
# This script takes a directory of input files and partitions them into a list of
# input files to use for TRAINING, VALIDATION, and TESTING 
#


# set our random seed based on current time
now = int(time())
seed(now)

# Set some paths for our image library of raw and binary labeled data
IMG_RAWPATH = "../images/raw"
IMG_BINPATH = "../images/binary"

# Set output filenames
TRAIN_FILENAME      = "train_list.txt"
VALIDATION_FILENAME = "validation_list.txt"
TEST_FILENAME    = "test_list.txt"

FILE_FILTER = "Po1g*.tif"

# The fraction of the image library that will be used for training, validation, and testing
# The totals must add up to 1.0
TRAIN_FRACTION      = 0.15
VALIDATION_FRACTION = 0.0
TEST_FRACTION       = 0.85

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
train_partition_end = int(floor(num_filenames*TRAIN_FRACTION-1))
validation_partition_size = int(floor(num_filenames*VALIDATION_FRACTION))
if(train_partition_end == num_filenames-1):
	test_partition_start = -1
	test_partition_end   = -1
	validation_partition_start = -1
	validation_partition_end = -1
else:
	validation_partition_start = train_partition_end+1
	validation_partition_end = validation_partition_start + validation_partition_size
	test_partition_start = validation_partition_end + 1
	test_partition_end   = num_filenames-1

if(train_partition_start < 0 or train_partition_end < 0 or train_partition_end < train_partition_start):
	train_partition_start = -1
	train_partiiton_end = -1
	

print("TOTAL NUMBER OF FILENAMES: %d" %num_filenames)
print("TRAIN: %d thru %d" %(train_partition_start,train_partition_end))
print("VALIDATION: %d thru %d" %(validation_partition_start,validation_partition_end))
print("TEST: %d thru %d" %(test_partition_start,test_partition_end))


# Write the training set input file
if(train_partition_start >= 0):
	file = open(TRAIN_FILENAME, "w")
	for i in range(train_partition_start, train_partition_end+1):
		file.write(filenames[i] + "\n")
	file.close()

# Write the validation set input file
if(validation_partition_start >= 0):
	file = open(VALIDATION_FILENAME, "w")
	for i in range(validation_partition_start, validation_partition_end+1):
		file.write(filenames[i] + "\n")
	file.close()

# Write the test set input file
if(test_partition_start >= 0):
	file = open(TEST_FILENAME, "w")
	for i in range(test_partition_start, test_partition_end+1):
		file.write(filenames[i] + "\n")
	file.close()

exit(0)


