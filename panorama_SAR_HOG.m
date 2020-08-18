clc;
close all;
clear all;
%% Step 1 - Load Images
% Load images.
% buildingDir = fullfile(toolboxdir('vision'), 'visiondata', 'building');%good
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\uttower');%good
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\pier');%good
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\ledge');%good
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\hill');%good
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\building');%good
buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\air\test');%***good
% buildingDir = fullfile('C:\Users\AUT\Desktop\951029\ForMrPourfard\test13');
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\wall');%good
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\landscape');%bad
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\landscape2');%good
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\river');%good
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\roof');%good
% buildingDir = fullfile('C:\Users\AUT\Desktop\951029\ForMrPourfard\test6');%large panorama
% buildingDir = fullfile('C:\Users\AUT\Desktop\951029\ForMrPourfard\test7');%bad%color and gray
% buildingDir = fullfile('C:\Users\AUT\Desktop\951029\ForMrPourfard\test10');
% buildingDir = fullfile('C:\Users\AUT\Desktop\951029\ForMrPourfard\test9');%bad
% buildingDir='D:\MATLAB\descriptors\stitch\Image-Stitching-master\Image-Stitching-master\img\air\test';
% buildingDir='C:\Users\AUT\Desktop\951029\ForMrPourfard\test3';
buildingScene = imageDatastore(buildingDir);
% Display images to be stitched
% figure(1),montage(buildingScene.Files);
%% Step 2 - Register Image Pairs
% Read the first image from the image set.
I = readimage(buildingScene, 1);

% Initialize features for I(1)
if(size(I,3)==3)
    grayImage = rgb2gray(I);
else
    grayImage=I;
end

% points = detectSURFFeatures(grayImage);
% [features, points] = extractFeatures(grayImage, points);

[features,points] = extractHOGFeatures(grayImage);

grayImage_prev=grayImage;
% Initialize all the transforms to the identity matrix. Note that the
% projective transform is used here because the building images are fairly
% close to the camera. Had the scene been captured from a further distance,
% an affine transform would suffice.
numImages = numel(buildingScene.Files);
tforms(numImages) = projective2d(eye(3));

% Iterate over remaining image pairs
for n = 2:numImages
    
    % Store points and features for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;
    
    % Read I(n).
    I = readimage(buildingScene, n);
    
    % Detect and extract SURF features for I(n).
    if(size(I,3)==3)
        grayImage = rgb2gray(I);
    else
        grayImage=I;
    end
%     points = detectSURFFeatures(grayImage);
%     [features, points] = extractFeatures(grayImage, points);
    
    [features,points] = extractHOGFeatures(grayImage);
    
    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);
    
    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);
    figure(2); showMatchedFeatures(grayImage,grayImage_prev,matchedPoints,matchedPointsPrev);
    legend('matched points 1','matched points 2');
     showMatchedFeatures_separate(grayImage,grayImage_prev,matchedPoints,matchedPointsPrev);
    legend('matched points 1','matched points 2');
    
    grayImage_prev=grayImage;
    % Estimate the transformation between I(n) and I(n-1).
    tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    
    % Compute T(1) * ... * T(n-1) * T(n)
    tforms(n).T = tforms(n-1).T * tforms(n).T;
    n
end
% At this point, all the transformations in tforms are relative to the first image. 
% This was a convenient way to code the image registration procedure because
% it allowed sequential processing of all the images. However, using the first 
% image as the start of the panorama does not produce the most aesthetically
% pleasing panorama because it tends to distort most of the images that form the panorama.
% A nicer panorama can be created by modifying the transformations such that 
% the center of the scene is the least distorted. This is accomplished by 
% inverting the transform for the center image and applying that transform 
% to all the others.
% Start by using the projective2d outputLimits method to find the output limits
% for each transform. The output limits are then used to automatically fin%good***d
%     the image that is roughly in the center of the scene.
imageSize = size(I);  % all the images are the same size%why????????????

% Compute the output limits  for each transform
for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
end
% Next, compute the average X limits for each transforms and find the image
% that is in the center. Only the X limits are used here because the scene
% is known to be horizontal. If another set of images are used, both the X 
% and Y limits may need to be used to find the center image.
avgXLim = mean(xlim, 2);

[~, idx] = sort(avgXLim);

centerIdx = floor((numel(tforms)+1)/2);

centerImageIdx = idx(centerIdx);
% Finally, apply the center image's inverse transform to all the others.
Tinv = invert(tforms(centerImageIdx));

for i = 1:numel(tforms)%all of the image are aligned with central image
    tforms(i).T = Tinv.T * tforms(i).T;
end
%% Step 3 - Initialize the Panorama
% Now, create an initial, empty, panorama into which all the images are mapped.
% Use the outputLimits method to compute the minimum and maximum output
% limits over all transformations. These values are used to automatically 
% compute the size of the panorama.

for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
end

% Find the minimum and maximum output limits
xMin = min([1; xlim(:)]);
xMax = max([imageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([imageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
if (size(I,3)==3)
panorama = zeros([height width 3], 'like', I);
else
    panorama = zeros([height width], 'like', I);
end

%% Step 4 - Create the Panorama
% Use imwarp to map images into the panorama and use vision.AlphaBlender 
% to overlay the images together.

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');
% blender = vision.AlphaBlender('Operation', 'Highlight selected pixel', ...
%     'MaskSource', 'Input port');

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama.
for i = 1:numImages
    
    I = readimage(buildingScene, i);
    
    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
    
    % Generate a binary mask.
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
    
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
    figure(3),imshow(panorama);
    pause(0.1);
end

% figure
% imshow(panorama);
