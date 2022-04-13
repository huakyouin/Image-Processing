clc;clear all;close all
img=imread('图片库//moon.tif');%读取图像信息
subplot(231)
imshow(img)
title('origin image')
b=fill(img);
b=laplacefliter(b);
subplot(232)
imshow(b)
title('laplace fliter')
img=im2double(img); 
subplot(233)
imshow(img-0.1*b)
title('sharpened with w=0.1')
subplot(234)
imshow(img-0.5*b)
title('sharpened with w=0.5')
subplot(235)
imshow(img-0.8*b)
title('sharpened with w=0.8')
subplot(236)
imshow(img-1*b)
title('sharpened with w=1')

% 扩展图像来处理边界
function B=fill(A) % 采用复制图像边界来扩展 
k=3; % 锐化滑窗为3
p=(k-1)/2; % 滑窗半径
[m,n,o]=size(A);
% 因为下面生成的扩展矩阵为double类型，为使矩阵乘法合法必须转化
A=im2double(A); 
% 扩展矩阵：temp2为行扩展，temp1为列扩展
temp1=blkdiag(ones(1,p+1),eye(n-2),ones(1,p+1));
temp2=blkdiag(ones(p+1,1),eye(m-2),ones(p+1,1));
% 对所有通道操作赋值
for i=1:o
    B(:,:,i)=temp2*A(:,:,i)*temp1;
end
end

% 获得拉普拉斯滤波
function b=laplacefliter(img)
k=3; % 锐化滑窗为3
p=(k-1)/2; % 滑窗半径
[m,n,o]=size(img);
% 因为下面生成的滤波器矩阵为double类型，为使矩阵乘法合法必须转化
img=im2double(img); 
% 原尺寸
M=m-2*p;
N=n-2*p;
b=zeros(M,N,o);
% 生成图像二阶导数滤波器
temp=zeros(k);
temp(p+1,1:k)=ones(1,k);
temp(1:k,p+1)=ones(k,1);
temp(p+1,p+1)=2-2*k;
% 进行卷积核操作
for i=1:M
    for j=1:N
        for t=1:o
        b(i,j,t)=sum(sum(temp.*img(i:i+2*p,j:j+2*p,t)));
        end
    end
end
% 归一化
for i=1:o
    maxb=max(max(b(:,:,i)));
    minb=min(min(b(:,:,i)));
    b(:,:,i)=(b(:,:,i)-minb)./(maxb-minb);
end
end