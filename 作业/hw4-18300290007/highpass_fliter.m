clc;clear all;close all;
A=imread('图片库//202153.jpg');%读取图像信息
if numel(size(A))>2 A=rgb2gray(A); end % 非灰则转灰
b=fill(A);
f=getf(b);
[m,n]=size(A);
h=geth(A,30);
c=cut((iget(f,h)));
subplot(231);
imshow(A);
title('原图');
subplot(232);
imshow(uint8(b));
title('填充后');
subplot(233);
imshow(h);
title('高斯滤波器σ=30');
subplot(234);
imshow(feq(f));
title('原图频谱图');
subplot(235)
imshow(A+3*c);
title('高通锐化操作结果（蒙版参数=3）');
subplot(236);
imshow(feq(getf(fill(A+3*c))));
title('操作后图的频谱图');
% 填充函数(零倍增）
function B=fill(A) 
[m,n]=size(A);
B=zeros(2*m,2*n);
B(1:m,1:n)=A;
end
% 变换函数
function B=getf(A)
[m,n]=size(A);
% 中心平移
for i=1:m
    for j=1:n
        A(i,j)=(-1)^(i+j)*A(i,j);
    end
end
% DFT
B=fft2(A);
end
% 谱函数--对变换后数组做对数处理，用于生成频谱图
function B=feq(f)
lgf=log(1+abs(f));
maxf=max(max(lgf));
minf=min(min(lgf));
B=(lgf-minf)/(maxf-minf);
end
% 滤波器函数（高斯高通）
function B=geth(A,sigma,c)
[m,n]=size(A);
B=zeros(2*m,2*n);
for x=1:2*m
    for y=1:2*n
        r2 = (x-m)^2 + (y-n)^2;
        B(x,y) =1- exp(-r2 / (2*sigma^2));
    end
end
end
% 逆变换函数 
function B=iget(f,h)
B=real(ifft2(f.*h));
[m,n]=size(B);
for x=1:m
    for y=1:n
        B(x,y)=B(x,y)*(-1)^(x+y);
    end
end
end
% 截取函数
function B=cut(A)
[m,n]=size(A);
B=A(1:m/2,1:n/2);
B=uint8(B);
end