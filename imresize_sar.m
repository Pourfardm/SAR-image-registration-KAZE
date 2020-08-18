
clear all;
close all;
k=0.25;
A=imresize(imread('C:\Users\hp\Pictures\SAR image\aks\scene2\3.jpeg'),k,'box');
imwrite(A,'C:\Users\hp\Pictures\SAR image\aks\scene2_new\2.jpg');
   
A=imresize(imread('C:\Users\hp\Pictures\SAR image\aks\scene2\4_1.jpeg'),k,'box');
imwrite(A,'C:\Users\hp\Pictures\SAR image\aks\scene2_new\1.jpg');

A=imresize(imread('C:\Users\hp\Pictures\SAR image\aks\scene2\4_2.jpeg'),k,'box');
imwrite(A,'C:\Users\hp\Pictures\SAR image\aks\scene2_new\3.jpg');
%% %%%%%%%%%%%%%%%%%%%
A=imresize(imread('C:\Users\hp\Pictures\SAR image\aks\scene1\1.jpeg'),k,'box');
imwrite(A,'C:\Users\hp\Pictures\SAR image\aks\scene1_new\1.jpg');

A=imresize(imread('C:\Users\hp\Pictures\SAR image\aks\scene1\2.jpeg'),k,'box');
imwrite(A,'C:\Users\hp\Pictures\SAR image\aks\scene1_new\2.jpg');