close all; clear all; clc;
testfiledir = './training-d/';
testfiledir1 = './training-f/';

img_file = dir(fullfile(testfiledir,'*.png'));
dir_len = length(img_file);
for i = 1:dir_len
    i
    img = gpuArray(imread(fullfile(testfiledir, img_file(i).name)));
    img_noise = imnoise(img,'salt & pepper',.08);
    img_noise = gather(img_noise);
    imwrite(img_noise, fullfile(testfiledir1, img_file(i).name));
    
end

