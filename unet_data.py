from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from random import shuffle
import numpy as np 
import os
import glob
import skimage.io as io
from skimage import img_as_ubyte
import skimage.transform as trans


def trainGenerator(batch_size,train_path,image_folder,mask_folder,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,target_size = (128,128),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''

    image_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_datagen = ImageDataGenerator()
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_generator = zip(image_generator, mask_generator)

    for (img,mask) in train_generator:
        yield (img/255,mask/255)



def testGenerator(testdir_path,filenames):

    for f in filenames:
        fullpath = testdir_path + '/' + f
        img = io.imread(fullpath,as_gray = True)
        img = img / 255
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img


def saveResult(save_path,filenames,results):
    for i,item in enumerate(results):
        img = item[:,:,0]
        bin_img = img > 0.5
        io.imsave(os.path.join(save_path,filenames[i]),img_as_ubyte(bin_img))
