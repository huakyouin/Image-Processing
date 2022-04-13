function c=myem(img,k)
img=double(img);
% 设置初始均值和方差，均值为随机在各色块中选点并取整所得
mu=rand(k,1)*255;
mu=sort(mu);
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