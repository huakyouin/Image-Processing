clc;clear all;close all;
img=imread('图片库//心脏图像.png');%读取图像信息
if numel(size(img))>2 img=rgb2gray(img); end % 非灰则转灰
% subplot(231)
% imshow(img);
% title('origin image');
% subplot(232)
% x=renyi(img,2,2000);
% imshow(x);
% title('renyi noise a=2,b=2000');
% subplot(233)
% x=renyi(img,100,2);
% imshow(x);
% title('renyi noise a=15,b=2');
% subplot(234)
% y=jiaoyan(img,100,-100,0.1,0.1);
% imshow(y);
% title('jiaoyan noise a=100,b=-100,pa=0.1,pb=0.1');
% subplot(235)
% y=jiaoyan(img,10,-10,0.4,0.4);
% imshow(y);
% title('jiaoyan noise a=10,b=-10,pa=0.4,pb=0.4');
% subplot(236)
% y=jiaoyan(img,100,-100,0.49,0.49);
% imshow(y);
% title('jiaoyan noise,a=100,b=-100,pa=0.49,pb=0.49');
y=renyi(img,2,2000);
imshow(y);
imwrite(y,'test518.jpg','jpg');
% 瑞利噪声

