% 椒盐噪声
function y=jiaoyannoise(img,a,b,pa,pb)
[m,n]=size(img);
noise=zeros(m,n); % 申请噪声矩阵空间
cdf=rand(m,n); % 生成01均匀分布的同规模矩阵
% 将每个元素按噪声pdf反变换等效按噪声分布生成噪声矩阵
noise(cdf<pa)=a;
noise(cdf>1-pb)=b;
% 叠加
y=uint8(double(img)+noise);
end