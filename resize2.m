k=0.75;
A=imresize(rgb2gray(imread('E:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\air\test\12_3.tif')),k,'Box');
B=imresize(rgb2gray(imread('E:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\air\test\2224.png')),k,'Box');
C=imresize(rgb2gray(imread('E:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\air\filt6.png')),k,'Box');

imwrite(A,'C:\Users\hp\Pictures\SAR image\image of papers\test\test37\1.tif');
imwrite(B,'C:\Users\hp\Pictures\SAR image\image of papers\test\test37\3.tif');
imwrite(C,'C:\Users\hp\Pictures\SAR image\image of papers\test\test37\2.tif');

u=0.5;
A=imresize(imread('E:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\air\com1ee.tif'),u,'Box');
B=imresize(imread('E:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\air\com2ee - Copy.tif'),u,'Box');
imwrite(A,'C:\Users\hp\Pictures\SAR image\image of papers\test\test38\1.tif');
imwrite(B,'C:\Users\hp\Pictures\SAR image\image of papers\test\test38\2.tif');
