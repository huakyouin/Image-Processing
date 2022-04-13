clc;clear all;close all;
img=imread("图片库//心脏图像.png");
if numel(size(img))>2 img=rgb2gray(img); end % 非灰则转灰
k=5; %分成几类在此设置
x=2;y=2;
[u,img2]=mykmeans(img,5);
subplot(x,y,1)
imshow(img)
title('origin')
subplot(x,y,3)
imshow(em(img,k));
title(sprintf('EM k=%d',k))
subplot(x,y,2)
imshow(img2)
title(sprintf('kmeans k=%d',k))
subplot(x,y,4)
imshow(em(img,3))
title(sprintf('EM k=%d',k))
if numel(size(img))>2 img=rgb2gray(img); end % 非灰则转灰
function [u,b]=mykmeans(img,k)
%-------------
% 随机初始化
u=rand(k,1)*255;
u=sort(u);
img=double(img); % 浮点化防止后面运算报错
[m,n]=size(img);
c=zeros(m,n); % 每个像素点的类记录矩阵
I=ones(m,n);  % 用于后续便捷表达的辅助矩阵
d=ones(k,1);  % 每一类的迭代差距
err=10;       % 初始差距
%--------------
%正式迭代
while(err>0.1)
%根据当前每类的中心界定像素点类型
    for i=1:m
        for j=1:n
            min=255;
            for r=1:k
                if abs(img(i,j)-u(r))<=min
                    min=abs(img(i,j)-u(r));
                    c(i,j)=r-1;
                end
            end
        end
    end
%计算每一类点的灰度期望作为下一迭代的类中心
    for r=1:k
        temp=u(r);
        tot=sum(sum(img(c==r-1)));
        con=sum(sum(I(c==r-1)));
        u(r)=tot/con;
        d(r)=abs(u(r)-temp);
    end
%以最大迭代距离作为本次误差
    err=max(d);
end
%转换成灰度图数据格式
b=uint8(c/(k-1)*255);
end
%修改成以kmeans中心作为初始均值
function c=em(img,k)
img=double(img);
% 设置初始均值和方差，均值为随机在各色块中选点并取整所得
[mu,~]=mykmeans(img,k);
sigma =100*ones(1,k);
[m,n]=size(img);
px = zeros(m,n,k);
%全局每个点为某一类的概率
mypi = rand(1,k);
mypi = mypi/sum(mypi);%将类概率归一化
%以迭代次数来作为停止的条件
stopiter = 20;
iter = 1;
while iter <= stopiter
    %----------E-------------------
    N=zeros(m,n,k);
    for i=1:k
        for x=1:m
            for y=1:n
                N(x,y,i)=normpdf(img(x,y),mu(i),sigma(i))*mypi(i);
            end
        end
    end
    N2=N;
    N2(N<0.00000001)=0.000000001;% 防止分母为0 生成一个无0近似阵
    for x=1:m
        for y=1:n
            sumk_N=sum(N2(x,y,:));
            px(x,y,:)=N(x,y,:)/sumk_N;
        end
    end
    %----------M---------------------
    %更新参数集
    sum_all_px=sum(sum(sum(px)));
    for i=1:k
        sum_xy_px=sum(sum(px(:,:,i)));
        mu(i)=sum(sum(px(:,:,i).*img))/sum_xy_px;
        sigma(i) =sqrt(sum(sum(px(:,:,i).*((img-mu(i)).^2)))/sum_xy_px);
        mypi(i)=sum_xy_px/sum_all_px;
    end
    iter=iter+1;
end
c=zeros(m,n);
for x=1:m
    for y=1:n
% 返回(x，y)处最大概率类型给c
        [~,c(x,y)]=max(px(x,y,:));
    end
end
c=uint8((c-1)/(k-1)*255);
end