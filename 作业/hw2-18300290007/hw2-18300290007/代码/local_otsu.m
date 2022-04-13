clear all;clc;close all;
img=imread("图片库//writting.tif");
subplot(2,2,1);
imshow(img);
title("origin image");
bin=makebin(img);
t=otsu_find(bin,0);
img2=partition(img,t);
subplot(2,2,2);
imshow(img2);
title("global otsu")
img3=partition(img,0);
subplot(2,2,3);
imshow(img3);
title("threshold = 0")
img4=local_otsu_do(img(1:200,1:300),5);
subplot(2,2,4);
imshow(img4);
title("local otsu with k=5")
% otsu寻找阈值的函数
function t=otsu_find(bin,k)
s_max =[0,0];
N=sum(bin(:));
i=0:255;    %递增行向量，用于算期望
for threshold=1:256
    u=0;
    n_0 = sum(bin(1:threshold));  % 阈值以下像素数
    n_1 = sum(bin(threshold:256));  % 阈值以上像素数
    % 两侧频率
    w_0 = n_0/N;
    w_1 = n_1/N;
    % 阈值下平均灰度
    if(n_0>0)
        u_0 = i(1:threshold)*bin(1:threshold)/n_0;
    else 
        u_0=0;   %考虑极端划分
    end
    % 阈值上平均灰度
    if(n_1>0)
        u_1 = i(threshold:256)*bin(threshold:256)/n_1;
    else 
        u_1=0;   %考虑极端划分
    end
    % 总平均灰度
    u = w_0*u_0 + w_1*u_1;
    % 类间方差
    Dbet2 = w_0*((u_0-u)^2) + w_1*((u_1-u)^2);
    % 跟先前最优比较(取绝对大者意味着右偏，二分时阈值需分配给左半边)
    if (Dbet2>s_max(2))
        s_max=[threshold-1,Dbet2];
    end
end

t=s_max(1);
%局部法需开启的修正模式：当选区中存在奇异点，则让更新点跟随大流实现减弱噪音干扰
if(k==1)   
    if(w_0>0.95||w_1>0.95)   
        if(w_0>0.5)
            t=0;
        else
            t=255;
        end
    end
end
end
% 生成bin的函数
function bin=makebin(img)
[m,n]=size(img);
bin=zeros(256,1);
for i=1:m
    for j=1:n
        bin(img(i,j)+1)=bin(img(i,j)+1)+1;
    end
end
end
% 根据阈值进行二分的函数
function img_new=partition(img,t)  % t为阈值
[m,n]=size(img);
img_new=zeros(m,n);
for i=1:m
    for j=1:n
        if(img(i,j)<=t)
            img_new(i,j)=0;
        else
            img_new(i,j)=255;
        end
    end
end
end
% 局部otsu的函数
function img_new=local_otsu_do(img,k) % k为滑窗大小，应为奇数
[m,n]=size(img);
img_new=zeros(m,n);
q=(k-1)/2;  % 滑窗半径
b=zeros(m+2*q,n+2*q); %扩张工作空间减少边界讨论
b(1+q:m+q,1+q:n+q)=img; %搬运原数组
% 柔化扩充部分边界
b(1:q,1+q:n+q)=flipud(img(2:q+1,1:n));
b(m+q+1:m+2*q,1+q:n+q)=flipud(img(m-q:m-1,1:n));
b(1+q:m+q,1:q)=fliplr(img(1:m,2:q+1));
b(1+q:m+q,1+n+q:n+2*q)=fliplr(img(1:m,n-q:n-1));
bin=makebin(b(1:1+2*q,1:1+2*q)); %初始化
flag=1; % 采用缝针式滑窗，flag用于判断当前行是左滑还是右滑以及更新bin
j=1+q;
% 直接对上次作业滑窗代码进行改写
for i=1+q:m+q
    t=otsu_find(bin,1);
    img_new(i-q,j-q)=partition(b(i,j),t);
    sym=(1-flag)/2;  %配合flag实现判断当前行是左滑还是右滑
    qqq=1;  %判断是不是一行的开始的标志，如果是则跳过一次更新
    for j=(sym*(2*q+1+n)+flag*(1+q)):flag:(sym*(2*q+1+n)+flag*(n+q))
        if(qqq==1)
            qqq=0;
            continue;
        end
        for x=i-q:i+q
            bin(b(x,j-flag*(q+1))+1)= bin(b(x,j-flag*(q+1))+1)-1;
            bin(b(x,j+flag*q)+1)= bin(b(x,j+flag*q)+1)+1;
        end
        t=otsu_find(bin,1);
        img_new(i-q,j-q)=partition(b(i,j),t);
    end
    flag=-1*flag;
    if(i~=m+q)
        for x=j-q:j+q
            bin(b(i-q,x)+1)= bin(b(i-q,x)+1)-1;
            bin(b(i+1+q,x)+1)= bin(b(i+q+1,x)+1)+1;
        end
    end
end
end