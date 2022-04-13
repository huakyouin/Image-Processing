function a=otsu(img)
% 统计图像信息到bin
[m,n]=size(img);
bin=zeros(256,1);
for i=1:m
    for j=1:n
        bin(img(i,j)+1)=bin(img(i,j)+1)+1;
    end
end
% 初始化
s_max =[0,0];
a=zeros(m,n);
N=m*n;  % 像素个数
i=0:255;    %递增行向量，用于期望的向量表示
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
for i=1:m
    for j=1:n
        if img(i,j)>t a(i,j)=255; end
    end
end
end