# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 12:03:39 2017

@author: A669593
"""

from PIL import Image
import os
from os import listdir
from os.path import isfile, join

mypath=''
img=1
def resizeSave(imgpath,savepath,imgSizes):
    global img
    imgpath=mypath+imgpath
    onlyfiles = [f for f in listdir(imgpath) if isfile(join(imgpath, f))]
    
    savepath=mypath+savepath 
    for imgsize in imgSizes:
        directory=savepath+str(imgsize)
        if not os.path.exists(directory):
            os.makedirs(directory)   
            
    for filepath in onlyfiles:   
        if filepath!='Thumbs.db':   
            print(filepath)
            with Image.open(imgpath+filepath).convert('RGB') as img:
                #fileSize=img.size
                for imgsize in imgSizes:
                    imgNew = img.resize((imgsize,imgsize), Image.ANTIALIAS)
                    img=imgNew
                    imgNew.save(savepath+str(imgsize)+"\\"+filepath) 
            

                
#resizeSave("ValidAll\\","ValidSmall_",[28,32,64,128,300,600])   
resizeSave("baseImages\\","Valid_",[32])  

alldirs=['Test','TestReal','GEN','Real']
for directory in alldirs:
    os.makedirs(directory+"_32") 


 
"""resizeSave("ValidAll\\","ValidSmall_",[600])   

resizeSave("InValidAll\\","InValidSmall_",[32,64,128,300])   
resizeSave("ValidIncub\\","ValidSmall_",[32,64,128,300])  
resizeSave("InValidIncub\\","InValidSmall_",[32,64,128,300]) 
 
resizeSave("Convert\\","InValidSmall_",[32,64,128,300])  

resizeSave("Convert2\\","InValidSmall_",[32,64,128,300])  
resizeSave("ValidInternet\\","ValidSmall_",[32,64,128,300])
resizeSave("OriginalInvalid\\","OriginalInvalid_",[300])
resizeSave("Incorrect_images\\","Incorrect_images_",[64])"""
#fs=list(set(fileSize))   
#pixel=[x[0]*x[1]/1000 for x in fileSize]   
#pixelSet=      list(set(pixel))   


