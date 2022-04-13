clear all;clc;close all;
img=imread("black.tif");
subplot(2,2,1);
imshow(img);
title('origin image')
subplot(2,2,2);
A=localhist(img,5);
imshow(A)
title('k=5');
subplot(2,2,3);
A=localhist(img,9);
imshow(A)
title('k=9');
subplot(2,2,4);
A=localhist(img,13);
imshow(A)
title('k=13');

function A=localhist(B,k)
[m,n]=size(B);
q=(k-1)/2; %中心到边界距离
A=zeros(m,n); %修改后另存矩阵
c=zeros(256,1); %灰度统计列
b=zeros(m+2*q,n+2*q); %扩张矩阵
b(1+q:m+q,1+q:n+q)=B;
b(1:q,1+q:n+q)=flipud(B(2:q+1,1:n));
b(m+q+1:m+2*q,1+q:n+q)=flipud(B(m-q:m-1,1:n));
b(1+q:m+q,1:q)=fliplr(B(1:m,2:q+1));
b(1+q:m+q,1+n+q:n+2*q)=fliplr(B(1:m,n-q:n-1));
%imshow(b);
for i=1:1+2*q
    for j=1:1+2*q
        c(b(i,j)+1)=c(b(i,j)+1)+1;
    end
end %初始化
flag=1;
j=1+q;
for i=1+q:m+q
    A(i-q,j-q)=sum(c(1:b(i,j)+1))/k^2;
    sym=(flag-1)/2;
    for j=(-1*sym*(2*q+1+n)+flag*(1+q)):flag:(-1*sym*(2*q+1+n)+flag*(n+q))
        for x=i-q:i+q
            c(b(x,j-q)+1)= c(b(x,j-q)+1)-flag;
            c(b(x,j+q)+1)= c(b(x,j+q)+1)+flag;
        end
        A(i-q,j-q)=sum(c(1:b(i,j)+1))/k^2;
    end
    flag=-1*flag;
    if(i~=m+q)
        for x=j-q:j+q
            c(b(i-q,x)+1)= c(b(i-q,x)+1)-1;
            c(b(i+q,x)+1)= c(b(i+q,x)+1)+1;
        end
    end
end
A=uint8(A*255);
end