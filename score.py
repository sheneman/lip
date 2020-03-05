import getopt
import yaml
import sys
import cv2
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy
import re
from pprint import pprint


pattern = re.compile(".*tif")


#################################################################################
#
# HANDLE Command line arguments
#
#
def usage():
	print("python score.py [ --help | --verbose | --config=<YAML config filename> ] ")

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
	configfile="score.yaml"

#################################################################################

#################################################################################
#
# Format and Example config YAML file:
#
# FORMAT:
# -------
#   bindir: <path to bin images>
#   inputdir: <input directory with classified images>
#
# EXAMPLE:
# --------
#   bindir:             "../images/binary"
#   inputdir:        	"./classified_images_directory"
#
#################################################################################

cf = open(configfile, "r")
config = yaml.load(cf, Loader=yaml.FullLoader)
#print("YAML CONFIG:")
#for c in config:
#        print("    [%s]:\"%s\"" %(c,config[c]))
#print("\n")
cf.close()


###############################################
#
# Iterate through all images and try all methods
#

# a list of arrays

header = "FILENAME,True Positives,False Positives,True Negatives,False Negatives,TM_CCORR_NORMED,DICE,Jaccard,F0.5,Precision,Recall"
print(header)

filelist = [f for f in listdir(config["inputdir"]) if isfile(join(config["inputdir"], f))]
scores = numpy.empty((len(filelist),10), dtype=numpy.float32)
file_index = 0
for f in filelist:
	x = 0

	if(pattern.match(f)):

		binary_fullpath = config["bindir"] + '/' + f
		output_fullpath = config["inputdir"] + '/' + f

		binary_img = Image.open(binary_fullpath)
		output_img = Image.open(output_fullpath)

		binary_imgarray = numpy.array(binary_img)
		output_imgarray = numpy.array(output_img)


		numrows=len(binary_imgarray)
		numcols=len(binary_imgarray[0])

		totalsize=numrows*numcols

		TP = 0 # true positives
		FP = 0 # false positives
		TN = 0 # true negatives
		FN = 0 # false negatives

		#pprint(binary_imgarray)
		#numpy.savetxt("foo.txt", binary_imgarray, fmt="%d");
		
		for i in range(numrows):
			for j in range(numcols):
				#print("bin[%d][%d] = %d, output[%d][%d] = %d" %(i,j,i,j,binary_imgarray[i][j], output_imgarray[i][j]))
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
		#print("FP = ", FP);
		#print("TN = ", TN);
		#print("FN = ", FN);

		scores[file_index][x] = TP; x+=1
		scores[file_index][x] = FP; x+=1
		scores[file_index][x] = TN; x+=1
		scores[file_index][x] = FN; x+=1

		# OpenCV2 Template Cross-Correlation Normalized (TM_CCORR_NORMED)
		tm_ccorr = cv2.matchTemplate(output_imgarray,binary_imgarray,cv2.TM_CCORR_NORMED)[0][0]
		scores[file_index][x] = tm_ccorr; x+=1
		
	
		#DICE
		numerator   = (2*TP)
		denominator = (2*TP+FP+FN)
		if (denominator == 0): 
			DICE = float('NaN')
		else:
			DICE = numerator/denominator
		scores[file_index][x] = DICE; x+=1



		#JACCARD
		numerator = TP
		denominator = (TP+FP+FN)
		if (denominator == 0): 
			JACCARD = float('NaN')
		else:
			JACCARD   = numerator/denominator
		scores[file_index][x] = JACCARD; x+=1


	
		#F05	
		numerator = (1.25*TP)
		denominator = (1.25*TP+FP+0.25*FN)
		if (denominator == 0): 
			F05 = float('NaN')
		else:
			F05 = numerator/denominator
		scores[file_index][x] = F05; x+=1



		#PRECISION
		numerator = TP
		denominator = (TP+FP)
		if (denominator == 0): 
			PRECISION = float('NaN')
		else:
			PRECISION = numerator/denominator
		scores[file_index][x] = PRECISION; x+=1



		#RECALL
		numerator = TP
		denominator = (TP+FN)
		if(denominator == 0):
			RECALL = float('NaN')
		else:
			RECALL = numerator/denominator
		scores[file_index][x] = RECALL; x+=1


		# output the row
		output = f + ',' + str(TP) + ',' + str(FP) + ',' + str(TN) + ',' + str(FN) + ',' + str(tm_ccorr) + ',' + str(DICE) + ',' + str(JACCARD) + ',' + str(F05) + ',' + str(PRECISION) + ',' + str(RECALL);
		print(output);

		file_index += 1

scores = scores[0:file_index][:]


print("\n\n")
print(" Median TM_CCORR: %f" %(numpy.median(scores[:,4])))
print("     Median DICE: %f" %(numpy.nanmedian(scores[:,5])))
print("  Median JACCARD: %f" %(numpy.nanmedian(scores[:,6])))
print("     Median F.05: %f" %(numpy.nanmedian(scores[:,7])))
print("Median PRECISION: %f" %(numpy.nanmedian(scores[:,8])))
print("   Median RECALL: %f" %(numpy.nanmedian(scores[:,9])))
print("   Mean TM_CCORR: %f" %(numpy.mean(scores[:,4])))
print("       Mean DICE: %f" %(numpy.nanmean(scores[:,5])))
print("    Mean JACCARD: %f" %(numpy.nanmean(scores[:,6])))
print("       Mean F.05: %f" %(numpy.nanmean(scores[:,7])))
print("  Mean PRECISION: %f" %(numpy.nanmean(scores[:,8])))
print("     Mean RECALL: %f" %(numpy.nanmean(scores[:,9])))

