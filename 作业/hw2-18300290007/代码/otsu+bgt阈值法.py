import matplotlib.pyplot as plt
from skimage import io

# 定义函数实现寻找otsu阈值
def otsu_threshold(img):
    rows,cols = img.shape
    N = rows*cols
    bin=[0 for x in range(256)]
    # 得到图片的灰度分布列
    for x in range(rows):
        for y in range(cols):
            bin[img[x,y]] = bin[img[x,y]]+1
    s_max = (0, -1)
    # 遍历寻找最优划分阈值
    for threshold in range(0,255):
        n_0 = sum(bin[:threshold])  # 阈值以下像素数
        n_1 = sum(bin[threshold:])   # 阈值以上像素数
        # 两侧频率
        w_0 = n_0/N
        w_1 = n_1/N
        # 阈值下平均灰度
        u_0 = sum([i*bin[i] for i in range(0, threshold)])/n_0 if n_0>0 else 0   #考虑极端划分
        # 阈值上平均灰度
        u_1 = sum([i*bin[i] for i in range(threshold, 256)])/n_1 if n_1>0 else 0 
        # 总平均灰度
        u = w_0*u_0 + w_1*u_1
        # 类间方差
        Dbet2 = w_0*((u_0-u)**2) + w_1*((u_1-u)**2)
        # 跟先前最优比较(取绝对大者意味着右偏，二分时阈值需分配给左半边)
        if Dbet2>s_max[1]:
            s_max=(threshold,Dbet2)
    return s_max[0]

# 定义函数实现寻找bgt阈值
def bgt_threshold(img):
    rows,cols = img.shape
    bin=[0 for x in range(256)]
    # 设定初始阈值
    threshold=123
    t=0
    # 得到图片的灰度分布列
    for x in range(rows):
        for y in range(cols):
            bin[img[x,y]] = bin[img[x,y]]+1 
    # 迭代至阈值收敛
    while True:
        t = threshold
        n_0 = sum(bin[:int(threshold)])  # 阈值以下像素数
        n_1 = sum(bin[int(threshold):])   # 阈值以上像素数
        # 阈值下平均灰度
        u_0 = sum([i*bin[i] for i in range(0, int(threshold))])/n_0 if n_0>0 else 0   #考虑极端划分
        # 阈值上平均灰度
        u_1 = sum([i*bin[i] for i in range(int(threshold), 256)])/n_1 if n_1>0 else 0 
        # 得到新阈值
        threshold = (u_0+u_1)/2
        if abs(t-threshold)<0.1:
            break
    return threshold

#主函数部分
path='图片库//finger.tif'
img = io.imread(path)
# 原始图像
plt.subplot(1,3,1)
io.imshow(img)
plt.title('origin image')
plt.axis("off")

# otsu法
threshold=otsu_threshold(img)
print("otsu_threshold is",threshold)
# 根据阈值二分
rows,cols = img.shape
for i in range(rows):
    for j in range(cols):
        img[i,j]=0 if img[i,j]<=threshold else 255
plt.subplot(1,3,2)
io.imshow(img)
plt.title('after otsu_threshold')
plt.axis("off")

# bgt法
img = io.imread(path)
threshold=bgt_threshold(img)
print("bgt_threshold is",threshold)
rows,cols = img.shape
# 根据阈值二分
for i in range(rows):
    for j in range(cols):
        img[i,j]=0 if img[i,j]<=threshold else 255
plt.subplot(1,3,3)
io.imshow(img)
plt.title('after bgt_threshold')
plt.axis("off")
# 展示
plt.show()