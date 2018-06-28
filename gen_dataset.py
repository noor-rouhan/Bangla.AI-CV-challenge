import numpy as np
import cv2
import os
import random


base_path = './input/'
print(os.getcwd())
input_folder = base_path+'training-d/'
output_folder = base_path+'training-h/'
os.mkdir(output_folder)
# os.chdir(base_path+input_folder+'/')
files = os.listdir(input_folder)
# print(files)
for img in files:
    i = cv2.imread(input_folder+img,0)
    (thresh, im_bw) = cv2.threshold(i, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    index = np.where(im_bw == 0)
    coordinates = zip(index[0], index[1])
    coordinates = list(coordinates)
    rand_cor = random.sample(coordinates,3)
    for cor in rand_cor:
        cv2.rectangle(i,(cor[1],cor[0]),(cor[1]+10,cor[0]+10),(0,0,0),-1)
        
    cv2.imwrite(output_folder+img, i)



# print(files)