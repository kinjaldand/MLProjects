# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:53:59 2019

@author: A669593
"""


from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import random
import numpy as np

mypath=''
pixelval=32
imgpath='InValid_'+str(pixelval)
onlyfiles = [f for f in listdir(imgpath) if isfile(join(imgpath, f))]

msk = np.random.rand(len(onlyfiles)) < 0.9



  
        
for flag,filepath in zip(msk,onlyfiles):   
    if filepath!='Thumbs.db':   
        print(filepath)
        if flag==False:
            destIn="Test"+"_"+str(pixelval)
            destVal="TestReal"+"_"+str(pixelval)
        else:
            destIn="GEN"+"_"+str(pixelval)
            destVal="Real"+"_"+str(pixelval)
            
        with Image.open(imgpath+"\\"+filepath).convert('RGB') as img:
            img.save(destIn+"\\"+filepath) 
        print(filepath)
        filepath=filepath.replace("COV_","").replace("_0.",".")
        print(filepath)
        newpath="Valid_"+str(pixelval)+"\\"+filepath
        print(newpath)
        with Image.open(newpath).convert('RGB') as img:
            img.save(destVal+"\\"+filepath) 
        
                
                
                
                
                
                
                
                
                
                