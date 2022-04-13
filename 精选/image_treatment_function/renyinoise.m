function x=renyinoise(img,a,b)
[m,n]=size(img);
noise=rand(m,n); % 生成01均匀分布的同规模矩阵
% 将每个元素按噪声pdf反变换等效按噪声分布生成噪声矩阵
noise=a+(-b.*log(1-noise)).^0.5; 
% 叠加
x=img+uint8(noise);
end
