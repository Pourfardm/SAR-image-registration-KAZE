

k=1;
theta=10;

A=imresize(rgb2gray(imread('C:\Users\hp\Pictures\SAR image\image of papers\sar test\test25\s7-3.jpg')),[300 501],'Box');
B=imresize(rgb2gray(imread('C:\Users\hp\Pictures\SAR image\image of papers\sar test\test25\s7-2.jpg')),k,'Box');
C=imresize(imrotate(rgb2gray(imread('C:\Users\hp\Pictures\SAR image\image of papers\sar test\test25\s7-3.jpg')),30),[300 501],'Box');
D=imrotate(rgb2gray(imread('C:\Users\hp\Pictures\SAR image\image of papers\sar test\test25\s7-2.jpg')),theta);
E=imresize(imrotate(rgb2gray(imread('C:\Users\hp\Pictures\SAR image\image of papers\sar test\test25\s7-2.jpg')),theta),[300 501],'Box');
F=imrotate(imresize(rgb2gray(imread('C:\Users\hp\Pictures\SAR image\image of papers\sar test\test25\s7-2.jpg')),[300 501],'Box'),theta);

%shera
imwrite(A,'C:\Users\hp\Pictures\SAR image\image of papers\sar test\test43\2.tif');
imwrite(B,'C:\Users\hp\Pictures\SAR image\image of papers\sar test\test43\1.tif');
%rotation
imwrite(B,'C:\Users\hp\Pictures\SAR image\image of papers\test\test45\1.tif');
imwrite(D,'C:\Users\hp\Pictures\SAR image\image of papers\test\test45\2.tif');

%shear and rotation
imwrite(B,'C:\Users\hp\Pictures\SAR image\image of papers\test\test46\1.tif');
imwrite(F,'C:\Users\hp\Pictures\SAR image\image of papers\test\test46\2.tif');