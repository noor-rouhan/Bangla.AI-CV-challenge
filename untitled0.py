# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:32:02 2018

@author: auri
"""

#### function
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def test_data(x,resize_dim=None,path=None):
        x_orig = []
        X=[] # initialize empty list for resized images
        img=cv2.imread(path,cv2.IMREAD_COLOR) # images loaded in color (BGR)
        #img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # cnahging colorspace to GRAY
        img=cv2.resize(img,(resize_dim,resize_dim),interpolation=cv2.INTER_AREA) # resize image to 28x28
        x_orig.append(img)
        x_orig = np.array(x_orig)
        #X.append(np.expand_dims(img,axis=2)) # expand image to 28x28x1 and append to the list.
        gaussian_3 = cv2.GaussianBlur(img, (9,9), 10.0) #unblur
        img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) #filter
        img = cv2.filter2D(img, -1, kernel)
        thresh = 200
        maxValue = 255
        #th, img = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY);
        ret,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        X.append(img) # expand image to 28x28x1 and append to the list
        # display progres
        X=np.array(X) # tranform list to numpy array
    
        
  #  if  path_label is None:

x_auga_test = test_data()