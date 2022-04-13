clear all;clc;close all;
img=imread("C:\Users\jxh0706\Desktop\moon.tif");
subplot(1,2,1);
imshow(img);
title('origin image')

[m,n]=size(img);
count=zeros(256,1);
for i=1:m
    for j=1:n
        count(img(i,j)+1)=count(img(i,j)+1)+1;
    end
end
count=count./(m*n);    % 归一化
for i=2:256
    count(i)=count(i-1)+count(i);
end
count=uint8(count*255);
for i=1:m
    for j=1:n
        img(i,j)=count(img(i,j)+1);
    end
end
subplot(1,2,2);
imshow(img)
title('after histogram equalization')