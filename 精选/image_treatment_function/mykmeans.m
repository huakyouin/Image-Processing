function b=mykmeans(img,k)
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