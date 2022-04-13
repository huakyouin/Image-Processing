clc;clear all;close all;
A=imread('图片库//shepplogan.PNG');%读取图像信息
if numel(size(A))>2 A=rgb2gray(A); end % 非灰则转灰
subplot(221)
imshow(A);
title('原图');
f=getf(fill(A));
fq=feq(f);
subplot(222);
imshow(fq);
title('原图频谱图');
axis on;
k=10;
for i=100:10:600
    f=cover(f,i,i,k);
    f=cover(f,i+400,i,k);
    f=cover(f,730+i,730+i,k);
end
for i=20:10:1000
    f=cover(f,i+400,i,k);
    f=cover(f,i,i+400,k);
end
subplot(224);
imshow(feq(f));
title('修改后频谱图');
axis on;
subplot(223)
c=cut(real(iget(f)));
imshow(c);
title('修改后图');
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
% 逆变换函数
function B=iget(f)
B=real(ifft2(f));
[m,n]=size(B);
for x=1:m
    for y=1:n
        B(x,y)=B(x,y)*(-1)^(x+y);
    end
end
end
% 谱函数--对变换后数组做对数处理，用于生成频谱图
function B=feq(f)
lgf=log(1+abs(f));
maxf=max(max(lgf));
minf=min(min(lgf));
B=(lgf-minf)/(maxf-minf);
end
% 截取函数
function B=cut(A)
[m,n]=size(A);
B=A(1:m/2,1:n/2);
B=uint8(B);
end
% 擦除函数
function B=cover(A,x,y,k)
B=A;
for i=x-k:x+k
    for j=y-k:y+k
        B(i,j)=2000;
    end
end
end