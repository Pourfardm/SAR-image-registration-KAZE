clc;
close all;
clear all;
zoom_factor=1;
%% Step 1 - Load Images
% Load images.
% buildingDir = fullfile(toolboxdir('vision'), 'visiondata', 'building');%good
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\uttower');%good
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\pier');%good
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\ledge');%good
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\hill');%good
buildingDir = fullfile('d:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\building');%good
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\air\test');%good***
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\wall');%good
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\landscape');%bad
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\landscape2');%good
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\river');%good
% buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\roof');%good
% buildingDir = fullfile('C:\Users\AUT\Desktop\951029\ForMrPourfard\test6');%large panorama
% buildingDir = fullfile('C:\Users\AUT\Desktop\951029\ForMrPourfard\test10');%dual images
% buildingDir = fullfile('C:\Users\AUT\Desktop\951029\ForMrPourfard\test7');%bad%color and gray
% buildingDir='D:\MATLAB\descriptors\stitch\Image-Stitching-master\Image-Stitching-master\img\air\test';
% buildingDir='C:\Users\AUT\Desktop\951029\ForMrPourfard\test3';
buildingScene = imageDatastore(buildingDir);
% Display images to be stitched
% figure(1),montage(buildingScene.Files);
%% Step 2 - Register Image Pairs
% Read the first image from the image set.
% I = readimage(buildingScene, 1);
I = imresize(readimage(buildingScene, 1),zoom_factor);

% Initialize features for I(1)
if(size(I,3)==3)
    grayImage = rgb2gray(I);
else
    grayImage=I;
end

%% get a specific area
im1f=figure; imshow(grayImage);
figure(im1f), [x1,y1]=getpts
grayImage_prev=grayImage(round(min(y1)):round(max(y1)),round(min(x1)):round(max(x1)));
imshow(grayImage_prev);
imwrite(grayImage_prev,'C:\Users\AUT\Desktop\951029\ForMrPourfard\test8\cropped.jpg');

points = detectSURFFeatures(grayImage_prev);
[features, points] = extractFeatures(grayImage_prev, points);


% buildingScene.Files(1)='cropped.jpg';
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
%     I = readimage(buildingScene, n);
    I = imresize(readimage(buildingScene, n),zoom_factor);
    
    % Detect and extract SURF features for I(n).
    if(size(I,3)==3)
        grayImage = rgb2gray(I);
    else
        grayImage=I;
    end
    im2f=figure; imshow(grayImage);
figure(im2f), [x2,y2]=getpts
grayImage_prev2=grayImage(round(min(y2)):round(max(y2)),round(min(x2)):round(max(x2)));
imshow(grayImage_prev2);
imwrite(grayImage_prev2,'C:\Users\AUT\Desktop\951029\ForMrPourfard\test8\cropped2.jpg');
    
    
    points = detectSURFFeatures(grayImage_prev2);
    [features, points] = extractFeatures(grayImage_prev2, points);
    
    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);
    
    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);
    figure(2); showMatchedFeatures(grayImage_prev2,grayImage_prev,matchedPoints,matchedPointsPrev);
    legend('matched points 1','matched points 2');
   showMatchedFeatures_separate(grayImage_prev2,grayImage_prev,matchedPoints,matchedPointsPrev);
    legend('matched points 1','matched points 2');
    
    grayImage_prev=grayImage_prev2;
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
% for each transform. The output limits are then used to automatically find
%     the image that is roughly in the center of the scene.
% imageSize = size(I);  % all the images are the same size%why????????????


% buildingDir = fullfile('C:\Users\AUT\Desktop\951029\ForMrPourfard\test8');%good
% buildingScene = imageDatastore(buildingDir);

% Compute the output limits  for each transform
for i = 1:numel(tforms)
    imageSize = size(imresize(readimage(buildingScene, i),zoom_factor));
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
    imageSize = size(imresize(readimage(buildingScene, i),zoom_factor));
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
panorama = zeros([height width 3], 'like', I);
% panorama = zeros([height width], 'like', I);

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
    
%     I = readimage(buildingScene, i);
    I = imresize(readimage(buildingScene, i),zoom_factor);
    
    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
    
    % Generate a binary mask.
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
    
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
    figure(5),imshow(panorama);
    pause(0.1);
end

% figure
% imshow(panorama);
