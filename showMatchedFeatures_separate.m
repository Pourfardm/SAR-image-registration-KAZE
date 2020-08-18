function showMatchedFeatures_separate(grayImage_prev2,grayImage_prev,matchedPoints,matchedPointsPrev)

im1=grayImage_prev2;
im2=grayImage_prev;
loc1=matchedPoints.Location;
loc2=matchedPointsPrev.Location;
% Create a new image showing the two images side by side.
im3 = appendimages(im1,im2);

% Show a figure with lines joining the accepted matches.
figure('Position', [100 100 size(im3,2) size(im3,1)]);
% colormap('gray');
imshow(im3);
hold on;
cols1 = size(im1,2);
for i = 1: size(loc1,1)
     line([loc1(i,1) loc2(i,1)+cols1], ...
         [loc1(i,2) loc2(i,2)], 'Color',[rand(),rand(),rand()]);%random color showing
%      pause(1);
%      pause();
end
hold off;
