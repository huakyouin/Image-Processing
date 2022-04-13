clear all;clc;close all;
img=imread("D:\VSproject\图片库\smy.jpg");
subplot(2,2,1);
imshow(img);
title('origin image')


[m,n,~]=size(img);
count=zeros(256,256);
for i=1:m
    for j=1:n
        count(img(i,j,1)+1,img(i,j,2)+1)=1+count(img(i,j,1)+1,img(i,j,2)+1);
    end
end
count1=zeros(256,256);
for i=1:256
    for j=1:256
        count1(i,j)=reallog(count(i,j));
    end
end

subplot(1,2,2);
x=0:255;
y=0:255;
imagesc(x,y,count1);
axis xy;
colorbar();
title('2d joint hist with ln(x) level');
set(gca,'Position',[0.6 0.38 0.3 0.4]) 

subplot(2,2,3);
mesh(x,y,count)
title('3d joint hist');
