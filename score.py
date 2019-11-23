from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy
import re
from pprint import pprint


BINARY_PATH = "./test/binary"
OUTPUT_PATH = "./test/output"


pattern = re.compile(".*tif")


###############################################
#
# Iterate through all images and try all methods
#

print("FILENAME,True Positives,False Positives,True Negatives,False Negatives,DICE,Jaccard,F0.5,Precision,Recall");

filelist = [f for f in listdir(BINARY_PATH) if isfile(join(BINARY_PATH, f))]
for f in filelist:
	
	if(pattern.match(f)):

#		print(f + ",",end="")

		binary_fullpath = BINARY_PATH + '/' + f
		output_fullpath = OUTPUT_PATH + '/' + f

		binary_img = Image.open(binary_fullpath)
		output_img = Image.open(output_fullpath)

		binary_imgarray = numpy.array(binary_img)
		output_imgarray = numpy.array(output_img)

		numrows=len(binary_imgarray)
		numcols=len(binary_imgarray[0])

		totalsize=numrows*numcols

		TP = 0 # true positives
		TN = 0 # true negatives
		FP = 0 # false positives
		FN = 0 # false negatives

		#pprint(binary_imgarray)
		#numpy.savetxt("foo.txt", binary_imgarray, fmt="%d");
		
		for i in range(numrows):
			for j in range(numcols):
				if(binary_imgarray[i][j] == 0 and output_imgarray[i][j]==0):
					TN=TN+1
				elif(binary_imgarray[i][j] == 255 and output_imgarray[i][j] == 255):
					TP=TP+1
				elif(binary_imgarray[i][j] == 0 and output_imgarray[i][j] == 255):
					FP=FP+1
				elif(binary_imgarray[i][j] == 255 and output_imgarray[i][j] == 0):
					FN=FN+1
				else:
					print("ERROR Scoring file")
					exit(0)

		# For Debugging
		#print("TP = ", TP);
		#print("TN = ", TN);
		#print("FP = ", FP);
		#print("FN = ", FN);
	
		#DICE
		numerator   = (2*TP)
		denominator = (2*TP+FP+FN)
		if (denominator == 0): 
			DICE = float('NaN')
		else:
			DICE = numerator/denominator



		#JACCARD
		numerator = TP
		denominator = (TP+FP+FN)
		if (denominator == 0): 
			JACCARD = float('NaN')
		else:
			JACCARD   = numerator/denominator


	
		#F05	
		numerator = (1.25*TP)
		denominator = (1.25*TP+FP+0.25*FN)
		if (denominator == 0): 
			F05 = float('NaN')
		else:
			F05 = numerator/denominator



		#PRECISION
		numerator = TP
		denominator = (TP+FP)
		if (denominator == 0): 
			PRECISION = float('NaN')
		else:
			PRECISION = numerator/denominator



		#RECALL
		numerator = TP
		denominator = (TP+FN)
		if(denominator == 0):
			RECALL = float('NaN')
		else:
			RECALL = numerator/denominator


		# output the row
		output = f + ',' + str(TP) + ',' + str(FP) + ',' + str(TN) + ',' + str(FN) + ',' + str(DICE) + ',' + str(JACCARD) + ',' + str(F05) + ',' + str(PRECISION) + ',' + str(RECALL);
		print(output);
