from math import ceil
from time import sleep, time
from datetime import datetime
from random import seed
from random import random
import tensorflow as tf
import os
import numpy

from unet_model import *
from unet_data import *


# set our random seed based on current time
now = int(time())

seed(now)
numpy.random.seed(now)
tf.random.set_seed(now)
os.environ['PYTHONHASHSEED'] = str(now)

# Some basic training parameters
EPOCHS = 100
BATCH_SIZE = 25
TRAIN_SIZE = 5000
STEPS_PER_EPOCH = ceil(TRAIN_SIZE/BATCH_SIZE)



lipid_train = trainGenerator(BATCH_SIZE,'/nethome/sheneman/src/lip/experiment/unet/train','raw8','mask',seed=now)

model = unet()
model_checkpoint = ModelCheckpoint('unet_lipid_checkpoint.h5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(lipid_train,steps_per_epoch=STEPS_PER_EPOCH,epochs=EPOCHS,callbacks=[model_checkpoint])

model.save("unet.model.h5")



#testdir_path = "/nethome/sheneman/src/lip/experiment/unet/test/raw8"
#filenames = os.listdir(testdir_path)
#
#testGen = testGenerator(testdir_path, filenames)
#results = model.predict_generator(testGen,len(filenames),verbose=1)
#saveResult("./output",filenames,results)


