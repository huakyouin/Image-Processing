clc;clear all;close all;
img=imread('图片库//大脑图像.png');%读取图像信息
if numel(size(img))>2 img=rgb2gray(img); end % 非灰则转灰
subplot(231)
imshow(img);
title('origin image');
subplot(232)
x=renyi(img,2,2000);
imshow(x);
title('renyi noise a=2,b=2000');
subplot(233)
x=renyi(img,100,2);
imshow(x);
title('renyi noise a=15,b=2');
subplot(234)
y=jiaoyan(img,100,-100,0.1,0.1);
imshow(y);
title('jiaoyan noise a=100,b=-100,pa=0.1,pb=0.1');
subplot(235)
y=jiaoyan(img,10,-10,0.4,0.4);
imshow(y);
title('jiaoyan noise a=10,b=-10,pa=0.4,pb=0.4');
subplot(236)
y=jiaoyan(img,100,-100,0.49,0.49);
imshow(y);
title('jiaoyan noise,a=100,b=-100,pa=0.49,pb=0.49');
% 瑞利噪声
function x=renyi(img,a,b)
[m,n]=size(img);
noise=rand(m,n); % 生成01均匀分布的同规模矩阵
% 将每个元素按噪声pdf反变换等效按噪声分布生成噪声矩阵
noise=a+(-b.*log(1-noise)).^0.5; 
% 叠加
x=img+uint8(noise);
end
% 椒盐噪声
function y=jiaoyan(img,a,b,pa,pb)
[m,n]=size(img);
noise=zeros(m,n); % 申请噪声矩阵空间
cdf=rand(m,n); % 生成01均匀分布的同规模矩阵
% 将每个元素按噪声pdf反变换等效按噪声分布生成噪声矩阵
noise(cdf<pa)=a;
noise(cdf>1-pb)=b;
% 叠加
y=uint8(double(img)+noise);
end
