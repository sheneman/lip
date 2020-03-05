from unet_model import *
from unet_data import *


model = load_model("unet_lipid_checkpoint.h5")
#model = load_model("unet.model.h5")

testdir_path = "/nethome/sheneman/src/lip/experiment/unet/test/raw8"
filenames = os.listdir(testdir_path)

testGen = testGenerator(testdir_path, filenames)
results = model.predict_generator(testGen,len(filenames),verbose=1)
#results = model.predict_generator(testGen,20,verbose=1)
saveResult("./output",filenames,results)


