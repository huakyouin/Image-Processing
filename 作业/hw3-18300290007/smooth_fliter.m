clc;clear all;close all
img=imread('图片库//writting.tif');%读取图像信息
subplot(131)
imshow(img)
title('origin image')
b=fill(img,7);
b=smoothfliter(b,7);
subplot(132)
imshow(b)
title('smooth fliter with k=7')
b=fill(img,13);
b=smoothfliter(b,13);
subplot(133)
imshow(b)
title('smooth fliter with k=13')

% 扩展图像来处理边界
function B=fill(A,k) % 采用复制图像边界来扩展 
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

% 平滑操作函数
function b=smoothfliter(img,k)
p=(k-1)/2; % 滑窗半径
[m,n,o]=size(img);
% 因为下面生成的滤波器矩阵为double类型，为使矩阵乘法合法必须转化
img=im2double(img); 
% 原尺寸
M=m-2*p;
N=n-2*p;
b=zeros(M,N,o);
%生成平滑滤波器
temp=ones(k)/9;
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