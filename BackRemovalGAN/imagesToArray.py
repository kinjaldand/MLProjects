# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:17:24 2019

@author: A669593
"""


from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pickle

alldirs=['Test','TestReal','GEN','Real']
pixelval=32
channels=3
image_height=32
image_width=32

for folder in alldirs:
    folder=folder+"_"+str(pixelval)
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    print(folder,len(onlyfiles))
    dataset = np.ndarray(shape=(len(onlyfiles), image_height, image_width,channels),dtype=np.float32)
    for i,_file in enumerate(onlyfiles):
        img = load_img(folder + "/" + _file)  # this is a PIL image
        x = img_to_array(img)    
        x = (x - 127.5) / 127.5     #normalize
        dataset[i] = x
      
    with open(folder+".pkl","wb") as f:    
        pickle.dump(dataset,f)
        
    