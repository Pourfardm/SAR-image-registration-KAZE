clc;
close all;
clear all;
MinDiversity=0.7;%0.7
MaxVariation=0.25;%0.2
Delta=1;%10
BrightOnDark=0;
DarkOnBright=1;
MaxArea=0.75;
MinArea=30 ;%3  %/numel(I)

count=1;

    close all;
%     GG=imrotate(imread('C:\Users\HP\Pictures\Database Images\Our SAR database test\test28\2016_Zhu_Page_4_Image_0004.jpg'),theta);
%     imwrite(GG,'C:\Users\HP\Pictures\Database Images\Our SAR database test\test_rotation\2016_Zhu_Page_4_Image_0004_rotated.jpg');
    imwrite(imread('C:\Users\HP\Pictures\Database Images\Our SAR database test\test12\s2_1.tif'),'C:\Users\HP\Pictures\Database Images\Our SAR database test\test_rotation1\s2_1.tif');
    imwrite(imread('C:\Users\HP\Pictures\Database Images\Our SAR database test\test28\2016_Zhu_Page_4_Image_0004.jpg'),'C:\Users\HP\Pictures\Database Images\Our SAR database test\test_rotation1\2016_Zhu_Page_4_Image_0004_rotated.jpg');
%     GG=imrotate(imread('C:\Users\HP\Pictures\Database Images\Our SAR database test\test12\s2_1.tif'),theta);
%     GG=imresize(imread('C:\Users\HP\Pictures\Database Images\Our SAR database test\test12\s2-1.tif'),scale);
%     imwrite(GG,'C:\Users\HP\Pictures\Database Images\Our SAR database test\test_rotation1\s2-1_rotated.tif');
%     imshow(GG);title(['Theta= ',num2str(theta)]);
%     imshow(GG);title(['Scale= ',num2str(scale)]);
    
    
    %% Step 1 - Load Images
    % Load images.
    % buildingDir = fullfile(toolboxdir('vision'), 'visiondata', 'building');%good
    % buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\uttower');%good
    % buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\pier');%good
    % buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\ledge');%good
    % buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\hill');%good
    % buildingDir = fullfile('E:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\building');%good
    buildingDir = fullfile('C:\Users\HP\Pictures\Database Images\Our SAR database test\test12');%12
%     buildingDir = fullfile('C:\Users\HP\Pictures\Database Images\Our SAR database test\test_rotation1');
    % buildingDir = fullfile('C:\Users\HP\Pictures\Database Images\Our SAR database test\test14');%30
    % buildingDir = fullfile('C:\Users\hp\Pictures\SAR image\image of papers\test\test46');%13
    % buildingDir = fullfile('C:\Users\hp\Pictures\SAR image\aks\scene3_new');
    % buildingDir = fullfile('E:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\air\test1');%***good
    % buildingDir = fullfile('C:\Users\AUT\Desktop\951029\ForMrPourfard\test13');
    % buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\wall');%good
    % buildingDir = fullfile('E:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\landscape');%bad
    % buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\landscape2');%good
    % buildingDir = fullfile('D:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\river');%good
    % buildingDir = fullfile('E:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\roof');%good
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
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%% SURF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tStart1=tic;
    % points = detectHarrisFeatures(grayImage);
    % points = detectKAZEFeatures(grayImage);
    points = detectSURFFeatures(grayImage,'MetricThreshold',1000);
    %     [points cc] = detectMSERFeatures(grayImage);
    % [points cc] = detectMSERFeatures(grayImage,'ThresholdDelta' ,0.4,'RegionAreaRange',[30 14000],'MaxAreaVariation',0.7,'ROI',[1 1 size(grayImage,2) size(grayImage,1)] );
    % [points cc] = vl_mser_pourfard(grayImage,MinDiversity,MaxVariation,Delta,BrightOnDark,DarkOnBright,MaxArea,MinArea);
    
    % points = vl_sift_pourfard(grayImage);%????????
    % points = detectBRISKFeatures(grayImage);imshow(I); hold on; plot(points.selectStrongest(20));
    % points = detectFASTFeatures(grayImage);plot(corners.selectStrongest(50));
    
    
    [features_extracted, points_extracted] = extractFeatures(grayImage, points);
    grayImage_prev=grayImage;
    % Initialize all the transforms to the identity matrix. Note that the
    % projective transform is used here because the building images are fairly
    % close to the camera. Had the scene been captured from a further distance,
    % an affine transform would suffice.
    numImages = numel(buildingScene.Files);
    % tforms(numImages) = projective2d(eye(3));
    tforms(numImages) = affine2d(eye(3));
    numnum=[];
    
    % Iterate over remaining image pairs
    for n = 2:numImages
        
        % Store points and features for I(n-1).
        pointsPrevious_extracted = points_extracted;
        featuresPrevious_extracted = features_extracted;
        
        % Read I(n).
        I = readimage(buildingScene, n);
        
        % Detect and extract SURF features for I(n).
        if(size(I,3)==3)
            grayImage = rgb2gray(I);
        else
            grayImage=I;
        end
        
        %     points = detectHarrisFeatures(grayImage);
        %     points = detectKAZEFeatures(grayImage);
        points = detectSURFFeatures(grayImage,'MetricThreshold',1000);
        %     [points cc] = detectMSERFeatures(grayImage);
        %     [points cc] = detectMSERFeatures(grayImage,'ThresholdDelta' ,0.4,'RegionAreaRange',[30 14000],'MaxAreaVariation',0.7,'ROI',[1 1 size(grayImage,2) size(grayImage,1)] );
        %     [points cc] = vl_mser_pourfard(grayImage,MinDiversity,MaxVariation,Delta,BrightOnDark,DarkOnBright,MaxArea,MinArea);
        % points = detectBRISKFeatures(grayImage);%imshow(I); hold on; plot(points.selectStrongest(20));
        % points = detectFASTFeatures(grayImage);plot(corners.selectStrongest(50));
        
        [features_extracted, points_extracted] = extractFeatures(grayImage, points);
        
        % Find correspondences between I(n) and I(n-1).
        indexPairs = matchFeatures(features_extracted, featuresPrevious_extracted, 'Unique', true);
        matchedPoints = points_extracted(indexPairs(:,1), :);
        matchedPointsPrev = pointsPrevious_extracted(indexPairs(:,2), :);
        %     figure(2), showMatchedFeatures(grayImage,grayImage_prev,matchedPoints,matchedPointsPrev);
        %     legend('matched points 1','matched points 2');
        %     title(['Mixed Matching images number: ',num2str(n),' to image number: ',num2str(n-1)]); drawnow;
        %     showMatchedFeatures_separate(grayImage,grayImage_prev,matchedPoints,matchedPointsPrev);
        %     legend('matched points 1','matched points 2');
        %     title(['Separate Matching images number: ',num2str(n),' to image number: ',num2str(n-1)]); drawnow;
        
        grayImage_prev=grayImage;
        % Estimate the transformation between I(n) and I(n-1).
        %     tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
        %         'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
        [tforms(n),inlierpoints1,inlierpoints2] = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
            'affine','Confidence', 99.9, 'MaxNumTrials', 2000);
        
        % Compute T(1) * ... * T(n-1) * T(n)
        tforms(n).T = tforms(n-1).T * tforms(n).T;
        
        precision=inlierpoints1.Count/numel(indexPairs(:,1))*100;
        recall=inlierpoints1.Count/[numel(features_extracted(:,1))+numel(featuresPrevious_extracted(:,1))-numel(indexPairs(:,1))]*100;
        f_measure=2*precision*recall/(precision+recall);
        numnum=[numnum;[numel(features_extracted(:,1)) , numel(featuresPrevious_extracted(:,1)), ...
            numel(indexPairs(:,1)),...
            inlierpoints1.Count,...
            precision,...
            recall,...
            f_measure,...
            toc(tStart1)
            ]];
        n;
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
    % blender = vision.AlphaBlender('Operation', 'Blend', ...
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
        %     figure(i+4),imshow(panorama);title(['Connecting images, now image number: ',num2str(i)]);
        %     figure(10),subplot(3,1,i),imshow(panorama);title(['Connecting images, now image number: ',num2str(i)]);
        %     pause(1);
    end
    
    
%     figure
%     imshow(panorama);title('SURF Result');
%     xlabel(['Theta= ',num2str(theta)]);
% %     xlabel(['Scale= ',num2str(scale)]);
%     pause(0.5);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MSER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % clear points,features_extracted,points_extracted,pointsPrevious_extracted,...
    %     featuresPrevious_extracted;
    I = readimage(buildingScene, 1);
    % Initialize features for I(1)
    if(size(I,3)==3)
        grayImage = rgb2gray(I);
    else
        grayImage=I;
    end
    
    tStart2=tic;
    % points = detectHarrisFeatures(grayImage);
    % points = detectKAZEFeatures(grayImage);
    % points = detectSURFFeatures(grayImage,'MetricThreshold',1000);
    [points cc] = detectMSERFeatures(grayImage);
    % [points cc] = detectMSERFeatures(grayImage,'ThresholdDelta' ,0.4,'RegionAreaRange',[30 14000],'MaxAreaVariation',0.7,'ROI',[1 1 size(grayImage,2) size(grayImage,1)] );
    % [points cc] = vl_mser_pourfard(grayImage,MinDiversity,MaxVariation,Delta,BrightOnDark,DarkOnBright,MaxArea,MinArea);
    
    % points = vl_sift_pourfard(grayImage);%????????
    % points = detectBRISKFeatures(grayImage);imshow(I); hold on; plot(points.selectStrongest(20));
    % points = detectFASTFeatures(grayImage);plot(corners.selectStrongest(50));
    
    
    [features_extracted, points_extracted] = extractFeatures(grayImage, points);
    grayImage_prev=grayImage;
    % Initialize all the transforms to the identity matrix. Note that the
    % projective transform is used here because the building images are fairly
    % close to the camera. Had the scene been captured from a further distance,
    % an affine transform would suffice.
    numImages = numel(buildingScene.Files);
    % tforms(numImages) = projective2d(eye(3));
    tforms(numImages) = affine2d(eye(3));
    % numnum=[];
    
    % Iterate over remaining image pairs
    for n = 2:numImages
        
        % Store points and features for I(n-1).
        pointsPrevious_extracted = points_extracted;
        featuresPrevious_extracted = features_extracted;
        
        % Read I(n).
        I = readimage(buildingScene, n);
        
        % Detect and extract SURF features for I(n).
        if(size(I,3)==3)
            grayImage = rgb2gray(I);
        else
            grayImage=I;
        end
        
        %     points = detectHarrisFeatures(grayImage);
        %     points = detectKAZEFeatures(grayImage);
        %     points = detectSURFFeatures(grayImage,'MetricThreshold',1000);
        [points cc] = detectMSERFeatures(grayImage);
        %     [points cc] = detectMSERFeatures(grayImage,'ThresholdDelta' ,0.4,'RegionAreaRange',[30 14000],'MaxAreaVariation',0.7,'ROI',[1 1 size(grayImage,2) size(grayImage,1)] );
        %     [points cc] = vl_mser_pourfard(grayImage,MinDiversity,MaxVariation,Delta,BrightOnDark,DarkOnBright,MaxArea,MinArea);
        % points = detectBRISKFeatures(grayImage);%imshow(I); hold on; plot(points.selectStrongest(20));
        % points = detectFASTFeatures(grayImage);plot(corners.selectStrongest(50));
        
        [features_extracted, points_extracted] = extractFeatures(grayImage, points);
        
        % Find correspondences between I(n) and I(n-1).
        indexPairs = matchFeatures(features_extracted, featuresPrevious_extracted, 'Unique', true);
        matchedPoints = points_extracted(indexPairs(:,1), :);
        matchedPointsPrev = pointsPrevious_extracted(indexPairs(:,2), :);
        %     figure(2), showMatchedFeatures(grayImage,grayImage_prev,matchedPoints,matchedPointsPrev);
        %     legend('matched points 1','matched points 2');
        %     title(['Mixed Matching images number: ',num2str(n),' to image number: ',num2str(n-1)]); drawnow;
        %     showMatchedFeatures_separate(grayImage,grayImage_prev,matchedPoints,matchedPointsPrev);
        %     legend('matched points 1','matched points 2');
        %     title(['Separate Matching images number: ',num2str(n),' to image number: ',num2str(n-1)]); drawnow;
        
        grayImage_prev=grayImage;
        % Estimate the transformation between I(n) and I(n-1).
        %     tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
        %         'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
        [tforms(n),inlierpoints1,inlierpoints2] = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
            'affine','Confidence', 99.9, 'MaxNumTrials', 2000);
        
        % Compute T(1) * ... * T(n-1) * T(n)
        tforms(n).T = tforms(n-1).T * tforms(n).T;
        
        precision=inlierpoints1.Count/numel(indexPairs(:,1))*100;
        recall=inlierpoints1.Count/[numel(features_extracted(:,1))+numel(featuresPrevious_extracted(:,1))-numel(indexPairs(:,1))]*100;
        f_measure=2*precision*recall/(precision+recall);
        numnum=[numnum;[numel(features_extracted(:,1)) , numel(featuresPrevious_extracted(:,1)), ...
            numel(indexPairs(:,1)),...
            inlierpoints1.Count,...
            precision,...
            recall,...
            f_measure,...
            toc(tStart2)
            ]];
        n;
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
    % blender = vision.AlphaBlender('Operation', 'Blend', ...
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
        %     figure(i+4),imshow(panorama);title(['Connecting images, now image number: ',num2str(i)]);
        %     figure(10),subplot(3,1,i),imshow(panorama);title(['Connecting images, now image number: ',num2str(i)]);
        %     pause(1);
    end
    
%     figure
%     imshow(panorama);title('MSER Result');
% %     xlabel(['Scale= ',num2str(scale)]);
%     xlabel(['Theta= ',num2str(theta)]);
%     pause(0.5);
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%% BRISK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % I = readimage(buildingScene, 1);
    %
    % % Initialize features for I(1)
    % if(size(I,3)==3)
    %     grayImage = rgb2gray(I);
    % else
    %     grayImage=I;
    % end
    %
    % tStart3=tic;
    % % points = detectHarrisFeatures(grayImage);
    % % points = detectKAZEFeatures(grayImage);
    % % points = detectSURFFeatures(grayImage,'MetricThreshold',1000);
    % %     [points cc] = detectMSERFeatures(grayImage);
    % % [points cc] = detectMSERFeatures(grayImage,'ThresholdDelta' ,0.4,'RegionAreaRange',[30 14000],'MaxAreaVariation',0.7,'ROI',[1 1 size(grayImage,2) size(grayImage,1)] );
    % % [points cc] = vl_mser_pourfard(grayImage,MinDiversity,MaxVariation,Delta,BrightOnDark,DarkOnBright,MaxArea,MinArea);
    %
    % % points = vl_sift_pourfard(grayImage);%????????
    % points = detectBRISKFeatures(grayImage);%imshow(I); hold on; plot(points.selectStrongest(20));
    % % points = detectFASTFeatures(grayImage);plot(corners.selectStrongest(50));
    %
    %
    % [features_extracted, points_extracted] = extractFeatures(grayImage, points);
    % grayImage_prev=grayImage;
    % % Initialize all the transforms to the identity matrix. Note that the
    % % projective transform is used here because the building images are fairly
    % % close to the camera. Had the scene been captured from a further distance,
    % % an affine transform would suffice.
    % numImages = numel(buildingScene.Files);
    % % tforms(numImages) = projective2d(eye(3));
    % tforms(numImages) = affine2d(eye(3));
    % % numnum=[];
    %
    % % Iterate over remaining image pairs
    % for n = 2:numImages
    %
    %     % Store points and features for I(n-1).
    %     pointsPrevious_extracted = points_extracted;
    %     featuresPrevious_extracted = features_extracted;
    %
    %     % Read I(n).
    %     I = readimage(buildingScene, n);
    %
    %     % Detect and extract SURF features for I(n).
    %     if(size(I,3)==3)
    %         grayImage = rgb2gray(I);
    %     else
    %         grayImage=I;
    %     end
    %
    % %     points = detectHarrisFeatures(grayImage);
    % %     points = detectKAZEFeatures(grayImage);
    % %     points = detectSURFFeatures(grayImage,'MetricThreshold',1000);
    % %     [points cc] = detectMSERFeatures(grayImage);
    % %     [points cc] = detectMSERFeatures(grayImage,'ThresholdDelta' ,0.4,'RegionAreaRange',[30 14000],'MaxAreaVariation',0.7,'ROI',[1 1 size(grayImage,2) size(grayImage,1)] );
    % %     [points cc] = vl_mser_pourfard(grayImage,MinDiversity,MaxVariation,Delta,BrightOnDark,DarkOnBright,MaxArea,MinArea);
    % points = detectBRISKFeatures(grayImage);%imshow(I); hold on; plot(points.selectStrongest(20));
    % % points = detectFASTFeatures(grayImage);plot(corners.selectStrongest(50));
    %
    %     [features_extracted, points_extracted] = extractFeatures(grayImage, points);
    %
    %     % Find correspondences between I(n) and I(n-1).
    %     indexPairs = matchFeatures(features_extracted, featuresPrevious_extracted, 'Unique', true);
    %     matchedPoints = points_extracted(indexPairs(:,1), :);
    %     matchedPointsPrev = pointsPrevious_extracted(indexPairs(:,2), :);
    % %     figure(2), showMatchedFeatures(grayImage,grayImage_prev,matchedPoints,matchedPointsPrev);
    % %     legend('matched points 1','matched points 2');
    % %     title(['Mixed Matching images number: ',num2str(n),' to image number: ',num2str(n-1)]); drawnow;
    % %     showMatchedFeatures_separate(grayImage,grayImage_prev,matchedPoints,matchedPointsPrev);
    % %     legend('matched points 1','matched points 2');
    % %     title(['Separate Matching images number: ',num2str(n),' to image number: ',num2str(n-1)]); drawnow;
    %
    %     grayImage_prev=grayImage;
    %     % Estimate the transformation between I(n) and I(n-1).
    % %     tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
    % %         'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    %     [tforms(n),inlierpoints1,inlierpoints2] = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
    %         'affine','Confidence', 99.9, 'MaxNumTrials', 2000);
    %
    %     % Compute T(1) * ... * T(n-1) * T(n)
    %     tforms(n).T = tforms(n-1).T * tforms(n).T;
    %
    %
    %     precision=inlierpoints1.Count/numel(indexPairs(:,1))*100;
    %     recall=inlierpoints1.Count/[numel(features_extracted.Features(:,1))+numel(featuresPrevious_extracted.Features(:,1))-numel(indexPairs(:,1))]*100;
    %     f_measure=2*precision*recall/(precision+recall);
    %     numnum=[numnum;[numel(features_extracted.Features(:,1)) , numel(featuresPrevious_extracted.Features(:,1)), ...
    %         numel(indexPairs(:,1)),...
    %         inlierpoints1.Count,...
    %         precision,...
    %         recall,...
    %         f_measure,...
    %         toc(tStart3)
    %         ]]
    %
    %
    %     n;
    % end
    % % At this point, all the transformations in tforms are relative to the first image.
    % % This was a convenient way to code the image registration procedure because
    % % it allowed sequential processing of all the images. However, using the first
    % % image as the start of the panorama does not produce the most aesthetically
    % % pleasing panorama because it tends to distort most of the images that form the panorama.
    % % A nicer panorama can be created by modifying the transformations such that
    % % the center of the scene is the least distorted. This is accomplished by
    % % inverting the transform for the center image and applying that transform
    % % to all the others.
    % % Start by using the projective2d outputLimits method to find the output limits
    % % for each transform. The output limits are then used to automatically fin%good***d
    % %     the image that is roughly in the center of the scene.
    % imageSize = size(I);  % all the images are the same size%why????????????
    %
    % % Compute the output limits  for each transform
    % for i = 1:numel(tforms)
    %     [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
    % end
    % % Next, compute the average X limits for each transforms and find the image
    % % that is in the center. Only the X limits are used here because the scene
    % % is known to be horizontal. If another set of images are used, both the X
    % % and Y limits may need to be used to find the center image.
    % avgXLim = mean(xlim, 2);
    %
    % [~, idx] = sort(avgXLim);
    %
    % centerIdx = floor((numel(tforms)+1)/2);
    %
    % centerImageIdx = idx(centerIdx);
    % % Finally, apply the center image's inverse transform to all the others.
    % Tinv = invert(tforms(centerImageIdx));
    %
    % for i = 1:numel(tforms)%all of the image are aligned with central image
    %     tforms(i).T = Tinv.T * tforms(i).T;
    % end
    % %% Step 3 - Initialize the Panorama
    % % Now, create an initial, empty, panorama into which all the images are mapped.
    % % Use the outputLimits method to compute the minimum and maximum output
    % % limits over all transformations. These values are used to automatically
    % % compute the size of the panorama.
    %
    % for i = 1:numel(tforms)
    %     [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
    % end
    %
    % % Find the minimum and maximum output limits
    % xMin = min([1; xlim(:)]);
    % xMax = max([imageSize(2); xlim(:)]);
    %
    % yMin = min([1; ylim(:)]);
    % yMax = max([imageSize(1); ylim(:)]);
    %
    % % Width and height of panorama.
    % width  = round(xMax - xMin);
    % height = round(yMax - yMin);
    %
    % % Initialize the "empty" panorama.
    % if (size(I,3)==3)
    % panorama = zeros([height width 3], 'like', I);
    % else
    %     panorama = zeros([height width], 'like', I);
    % end
    %
    % %% Step 4 - Create the Panorama
    % % Use imwarp to map images into the panorama and use vision.AlphaBlender
    % % to overlay the images together.
    %
    % blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    %     'MaskSource', 'Input port');
    % % blender = vision.AlphaBlender('Operation', 'Highlight selected pixel', ...
    % %     'MaskSource', 'Input port');
    % % blender = vision.AlphaBlender('Operation', 'Blend', ...
    % %     'MaskSource', 'Input port');
    %
    % % Create a 2-D spatial reference object defining the size of the panorama.
    % xLimits = [xMin xMax];
    % yLimits = [yMin yMax];
    % panoramaView = imref2d([height width], xLimits, yLimits);
    %
    % % Create the panorama.
    % for i = 1:numImages
    %
    %     I = readimage(buildingScene, i);
    %
    %     % Transform I into the panorama.
    %     warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
    %
    %     % Generate a binary mask.
    %     mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
    %
    %     % Overlay the warpedImage onto the panorama.
    %     panorama = step(blender, panorama, warpedImage, mask);
    % %     figure(i+4),imshow(panorama);title(['Connecting images, now image number: ',num2str(i)]);
    % %     figure(10),subplot(3,1,i),imshow(panorama);title(['Connecting images, now image number: ',num2str(i)]);
    % %     pause(1);
    % end
    %
    % % figure
    % % imshow(panorama);
    %
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Kaze %%%%%%%%%%%%%%%%%%%%%%
    I = readimage(buildingScene, 1);
    
    % Initialize features for I(1)
    if(size(I,3)==3)
        grayImage = rgb2gray(I);
    else
        grayImage=I;
    end
    
    tStart4=tic;
    % points = detectHarrisFeatures(grayImage);
    points = detectKAZEFeatures(grayImage);
    % points = detectSURFFeatures(grayImage,'MetricThreshold',1000);
    %     [points cc] = detectMSERFeatures(grayImage);
    % [points cc] = detectMSERFeatures(grayImage,'ThresholdDelta' ,0.4,'RegionAreaRange',[30 14000],'MaxAreaVariation',0.7,'ROI',[1 1 size(grayImage,2) size(grayImage,1)] );
    % [points cc] = vl_mser_pourfard(grayImage,MinDiversity,MaxVariation,Delta,BrightOnDark,DarkOnBright,MaxArea,MinArea);
    
    % points = vl_sift_pourfard(grayImage);%????????
    % points = detectBRISKFeatures(grayImage);imshow(I); hold on; plot(points.selectStrongest(20));
    % points = detectFASTFeatures(grayImage);plot(corners.selectStrongest(50));
    
    
    [features_extracted, points_extracted] = extractFeatures(grayImage, points);
    grayImage_prev=grayImage;
    % Initialize all the transforms to the identity matrix. Note that the
    % projective transform is used here because the building images are fairly
    % close to the camera. Had the scene been captured from a further distance,
    % an affine transform would suffice.
    numImages = numel(buildingScene.Files);
    % tforms(numImages) = projective2d(eye(3));
    tforms(numImages) = affine2d(eye(3));
    % numnum=[];
    
    % Iterate over remaining image pairs
    for n = 2:numImages
        
        % Store points and features for I(n-1).
        pointsPrevious_extracted = points_extracted;
        featuresPrevious_extracted = features_extracted;
        
        % Read I(n).
        I = readimage(buildingScene, n);
        
        % Detect and extract SURF features for I(n).
        if(size(I,3)==3)
            grayImage = rgb2gray(I);
        else
            grayImage=I;
        end
        
        %     points = detectHarrisFeatures(grayImage);
        points = detectKAZEFeatures(grayImage);
        %     points = detectSURFFeatures(grayImage,'MetricThreshold',1000);
        %     [points cc] = detectMSERFeatures(grayImage);
        %     [points cc] = detectMSERFeatures(grayImage,'ThresholdDelta' ,0.4,'RegionAreaRange',[30 14000],'MaxAreaVariation',0.7,'ROI',[1 1 size(grayImage,2) size(grayImage,1)] );
        %     [points cc] = vl_mser_pourfard(grayImage,MinDiversity,MaxVariation,Delta,BrightOnDark,DarkOnBright,MaxArea,MinArea);
        % points = detectBRISKFeatures(grayImage);%imshow(I); hold on; plot(points.selectStrongest(20));
        % points = detectFASTFeatures(grayImage);plot(corners.selectStrongest(50));
        
        [features_extracted, points_extracted] = extractFeatures(grayImage, points);
        
        % Find correspondences between I(n) and I(n-1).
        indexPairs = matchFeatures(features_extracted, featuresPrevious_extracted, 'Unique', true);
        matchedPoints = points_extracted(indexPairs(:,1), :);
        matchedPointsPrev = pointsPrevious_extracted(indexPairs(:,2), :);
        %     figure(2), showMatchedFeatures(grayImage,grayImage_prev,matchedPoints,matchedPointsPrev);
        %     legend('matched points 1','matched points 2');
        %     title(['Mixed Matching images number: ',num2str(n),' to image number: ',num2str(n-1)]); drawnow;
        %     showMatchedFeatures_separate(grayImage,grayImage_prev,matchedPoints,matchedPointsPrev);
        %     legend('matched points 1','matched points 2');
        %     title(['Separate Matching images number: ',num2str(n),' to image number: ',num2str(n-1)]); drawnow;
        
        grayImage_prev=grayImage;
        % Estimate the transformation between I(n) and I(n-1).
        %     tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
        %         'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
        [tforms(n),inlierpoints1,inlierpoints2] = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
            'affine', 'Confidence', 99.9, 'MaxNumTrials', 2000);
        
        % Compute T(1) * ... * T(n-1) * T(n)
        tforms(n).T = tforms(n-1).T * tforms(n).T;
        
        precision=inlierpoints1.Count/numel(indexPairs(:,1))*100;
        recall=inlierpoints1.Count/[numel(features_extracted(:,1))+numel(featuresPrevious_extracted(:,1))-numel(indexPairs(:,1))]*100;
        f_measure=2*precision*recall/(precision+recall);
        numnum=[numnum;[numel(features_extracted(:,1)) , numel(featuresPrevious_extracted(:,1)), ...
            numel(indexPairs(:,1)),...
            inlierpoints1.Count,...
            precision,...
            recall,...
            f_measure,...
            toc(tStart4)
            ]];
        n;
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
    % blender = vision.AlphaBlender('Operation', 'Blend', ...
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
        %     figure(i+4),imshow(panorama);title(['Connecting images, now image number: ',num2str(i)]);
        %     figure(10),subplot(3,1,i),imshow(panorama);title(['Connecting images, now image number: ',num2str(i)]);
        %     pause(1);
    end
    
    
%     figure
%     imshow(panorama);title('KAZE Result');
% %     xlabel(['Scale= ',num2str(scale)]);
%     xlabel(['Theta= ',num2str(theta)]);
%     pause(0.5);
   numnum 
%     numnum3d(:,:,count)=numnum;
% 
% save panorama_sar_rotation.mat numnum3d grayImage;
% f_measure_all=shiftdim(shiftdim(numnum3d(:,7,:),1))%f-measure
% size(f_measure_all)
% X=repmat(10:10:170,[3,1])
% plot(X,f_measure_all');legend();

% count2 = timeseries(f_measure_all',0:10:180);%theta
% % count2 = timeseries(f_measure_all',1:-0.1:0.4);%scale
% count2.Name = 'F-measure';
% count2.TimeInfo.Units = 'Angle';
% figure,plot(count2,'-p','MarkerSize',6,'LineWidth',2);
% legend('SURF','MSER','KAZE');
