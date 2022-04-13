clc;clear all;close all
img=imread('图片库//test518.jpg');%读取图像信息
subplot(131)
imshow(img)
title('origin image')
b=fill(img,7);
b=mysmoothfliter(b,7);
subplot(132)
imshow(b)
title('smooth fliter with k=7')
b=fill(img,13);
b=mysmoothfliter(b,13);
subplot(133)
imshow(b)
title('smooth fliter with k=13')
imwrite(b,'test519.jpg','jpg')

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

