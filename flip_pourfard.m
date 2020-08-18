

% A=imread('E:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\roof\roofs2.jpg');
A=imread('C:\Users\hp\Pictures\SAR image\aks\scene2_new\1.jpg');
B = fliplr(A);
% imwrite(B,'E:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\roof\roofs3.jpg');
imwrite(B,'C:\Users\hp\Pictures\SAR image\aks\scene3_new\1.jpg');

A=imread('C:\Users\hp\Pictures\SAR image\aks\scene2_new\2.jpg');
imwrite(A,'C:\Users\hp\Pictures\SAR image\aks\scene3_new\2.jpg');

A=imread('C:\Users\hp\Pictures\SAR image\aks\scene2_new\3.jpg');
B = fliplr(A);
% imwrite(B,'E:\MATLAB\descriptors\stitch\imageStitchMatlab\imageStitchMatlab\img\roof\roofs3.jpg');
imwrite(B,'C:\Users\hp\Pictures\SAR image\aks\scene3_new\3.jpg');