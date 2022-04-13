clc;clear all;close all;
img=imread('图片库//yxy.jpg');%读取图像信息
% 可灰度化，下文代码兼容
% img=rgb2gray(img);  
subplot(121);
imshow(img);%显示原图
title('origin image');
subplot(122);
b=enlarge(img,7.5);
imshow(b);%显示处理后图
title('enlagred with k=7.5')
imwrite(b,'./图片库/双线性插值放大结果.png'); % 将图片保存到图片库

%进行双线性放大的函数
function b=enlarge(img,k)   % k为放大倍率，可以为非整数，但要大于一 
[m,n,c]=size(img);  % 记录原图数组三维
B=zeros(m+1,n+1,c); % 扩张原图数组省去（1）处讨论边界
B(1:m,1:n,:)=img;   % 搬运
% 生成目标图数组的工作空间
M=ceil(k*m); %向上取整
N=ceil(k*n);
b=zeros(M,N,c);
% 对目标图数组每个元素依此双线性插值
for i=1:M
    for j=1:N
        % 将新图坐标仿射变换到原图，(x,y)为插值所用点的左上点坐标，u、v为组合系数
        x=floor((i-1)*(m-1)/(M-1)+1);  
        y=floor((j-1)*(n-1)/(N-1)+1);
        u=(i-1)*(m-1)/(M-1)+1-x;
        v=(j-1)*(n-1)/(N-1)+1-y;
        b(i,j,:)=(1-v)*(1-u)*B(x,y,:)+v*(1-u)*B(x,y+1,:)+(1-v)*u*B(x+1,y,:)+u*v*B(x+1,y+1,:); % （1）
    end
end
b=uint8(b); % 整型化
end