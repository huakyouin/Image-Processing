clc;clear all;close all;
img=imread("图片库//test518.jpg");
if numel(size(img))>2 img=rgb2gray(img); end % 非灰则转灰
a1=2;a2=2;
subplot(a1,a2,1)
imshow(img)
title('origin')
subplot(a1,a2,2)
imshow(myem(img,2))
title('EM k=2')
subplot(a1,a2,3)
imshow(mykmeans(img,2))
title('kmeans k=2')
subplot(a1,a2,4)
imshow(otsu(img))
title('otsu')


