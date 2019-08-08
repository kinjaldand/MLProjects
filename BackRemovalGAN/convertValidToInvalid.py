# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:52:24 2017

@author: A669593
"""

import os
import numpy as np
#import pandas as pd
import scipy
#import sklearn
#from skimage import io
import math
from random import randint
from colour import Color

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p],p


def GetRangeColor(c,pixelrange):
    if (c>250).all():
        c[randint(0,2)]=240
    r1=np.maximum((c-3),0)/255
    r2=np.minimum((c+3),255)/255
    c1=Color(rgb=r1)
    c2=Color(rgb=r2)
    colors = list(c1.range_to(c2,pixelrange))
    rFinal=[x.rgb for x in colors]
    rFinal=np.array(rFinal)*255
    
    return rFinal.astype(int)
    

#mypath='C:\\Kinjal\\Work\\ICARD detection\\Fortum\\Images\\'
mypath=''    
pixelrange=32
Valid=os.listdir(mypath+"Valid_"+str(pixelrange))
filepathValid=mypath+"Valid_"+str(pixelrange)+"\\"
filepathSave=mypath+"InValid_"+str(pixelrange)+"\\"

if os.path.exists(filepathSave)==False:
    os.mkdir(filepathSave)
#bufferRange=list(range(235, 256))

N=9
iVal=235
imagesList=[]
label = []
rList=[1,80,81,160,161,255,210,255]
for i in Valid:
    print(i)
    rnd=[]
    image = scipy.misc.imread(filepathValid+i,mode='RGB')
    

    r1=[randint(rList[0],rList[5]) for _ in range(int(N/3))]
    r2=[randint(rList[0],rList[5]) for _ in range(int(N/3))]
    r3=[randint(rList[0],rList[5]) for _ in range(int(N/3))]
    #r4=[randint(rList[6],rList[7]) for _ in range(int(3))]
    r=np.array([r1,r2,r3]).reshape(3,3)
    
    r1=r[:,0]
    r2=r[:,1]
    r3=r[:,2]

    for i1 in r1:
        for i2 in r2:
            for i3 in r3:
                rnd.extend([i1,i2,i3])
                
    #rnd=[0,0,0]
    #rnd.extend(r4)  #extra light
    
    N2=int(len(rnd)/3)      
    rnd=np.array(rnd).reshape(N2,3)
    rndNew=[GetRangeColor(x,pixelrange) for x in rnd]
    
    multiImages=np.array([image for _ in range(N2)])

    #break
    for d1 in range(pixelrange):
        offset=0
        offset2=-1
        flag=0
        for d2_r in range(pixelrange):
            #d2=offset-d2_r if offset>0 else d2_r
            d2=offset-d2_r*offset2
            r=image[d1,d2,0]
            g=image[d1,d2,1]
            b=image[d1,d2,2]
            #if r in bufferRange and g in bufferRange and b in bufferRange:
            #if r == 255 and g == 255 and b == 255:
            if r >= iVal and g >= iVal and b >= iVal:
                for index in range(N2):
                    multiImages[index,d1,d2,0]=rndNew[index][d1][0]
                    multiImages[index,d1,d2,1]=rndNew[index][d1][1]
                    multiImages[index,d1,d2,2]=rndNew[index][d1][2]
            else:
                if flag==1:
                    break
                offset=2*pixelrange-(pixelrange-d2_r)
                offset2=1
                flag=1
                #print(str(r)+" "+str(g)+" "+str(b))
                #break
                
    N2=1   # to save only one image
    for index in range(N2):
        filename=i.split('.')
        scipy.misc.imsave(filepathSave+i,multiImages[index])
    #imagesList.append(image)
    #break
    


#N=10
#offset=0
#offset2=-1
#mid=6
#for i in range(N):
#  print(i,offset-i if offset>0 else i,offset-i*offset2)
#  if i==mid:
#      offset=2*N-(N-i)
#      offset2=1
#      print(offset)

    








