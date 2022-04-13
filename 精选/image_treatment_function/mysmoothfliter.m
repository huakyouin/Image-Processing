% 平滑操作函数
function b=mysmoothfliter(img,k)
p=(k-1)/2; % 滑窗半径
[m,n,o]=size(img);
% 因为下面生成的滤波器矩阵为double类型，为使矩阵乘法合法必须转化
img=im2double(img); 
% 原尺寸
M=m-2*p;
N=n-2*p;
b=zeros(M,N,o);
%生成平滑滤波器
temp=ones(k)/9;
% 进行卷积核操作
for i=1:M
    for j=1:N
        for t=1:o
        b(i,j,t)=sum(sum(temp.*img(i:i+2*p,j:j+2*p,t)));
        end
    end
end
% 归一化
for i=1:o
    maxb=max(max(b(:,:,i)));
    minb=min(min(b(:,:,i)));
    b(:,:,i)=(b(:,:,i)-minb)./(maxb-minb);
end
end