# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 23:33:43 2018

@author: auri
"""

base_path = './input/'
input_folder = base_path+'training-f/'
output_folder = base_path + 'training-k/'
import random
import cv2
import os
# os.mkdir('./input/training-i')
angles = [-45,-40,-30,-20,20,30,40,50]
files2=os.listdir(input_folder)
from PIL import Image
blur_cor = [7,10,15,20,22]
for img in files2:
    print(img)
 #   image = cv2.imread(input_folder+img)
#    blur = cv2.blur(image,(random.choice(blur_cor),random.choice(blur_cor)))
  #  cv2.imwrite(output_folder+img, blur)
    image = Image.open(input_folder+img)
    img2 = image.rotate(random.choice(angles))
    img2.save(output_folder+img)