clc;
close all;
clear all;
tic

%% Step 1 - Load Images
% Load images.
buildingDir = fullfile('C:\Users\HP\Pictures\Database Images\Our SAR database test\test28');%12
% buildingDir = fullfile('C:\Users\HP\Pictures\Database Images\Challenging optic images\roof');

buildingScene = imageDatastore(buildingDir);
% Display images to be stitched
% figure(1),montage(buildingScene.Files);
%% Step 2 - Register Image Pairs
% Read the first image from the image set.
Ia = readimage(buildingScene, 1);
Ib = readimage(buildingScene, 2);
% Initialize features for I(1)
if(size(Ia,3)==3)
    grayImage_a = rgb2gray(Ia);
else
    grayImage_a=Ia;
end

if(size(Ib,3)==3)
    grayImage_b = rgb2gray(Ib);
else
    grayImage_b=Ib;
end

tic

[fa,da] = vl_sift(im2single(grayImage_a)) ;
[fb,db] = vl_sift(im2single(grayImage_b)) ;
% F = VL_SIFT(I) computes the SIFT frames [1] (keypoints) F of the
%   image I. I is a gray-scale image in single precision. Each column
%   of F is a feature frame and has the format [X;Y;S;TH], where X,Y
%   is the (fractional) center of the frame, S is the scale and TH is
%   the orientation (in radians).

% [matches, scores] = vl_ubcmatch(da,db,1.5) ;
% [drop, perm] = sort(scores, 'descend') ;
% matches = matches(:, perm);
% scores  = scores(perm);
% [row1 col1]=ind2sub(size(da),matches(1,:));
% [row2 col2]=ind2sub(size(db),matches(2,:));

[matches1, scores1] = vl_ubcmatch(da,db,1.5) ;
[drop1, perm1] = sort(scores1, 'descend') ;
matches1 = matches1(:, perm1);
scores1  = scores1(perm1);
%   [MATCHES,SCORES] = VL_UBCMATCH(DESCR1, DESCR2) retuns the matches and
%   also the squared Euclidean distance between the matches.
[matches2, scores2] = vl_ubcmatch(db,da,1.5) ;
[drop2, perm2] = sort(scores2, 'descend') ;
matches2 = matches2(:, perm2);
scores2  = scores2(perm2);




