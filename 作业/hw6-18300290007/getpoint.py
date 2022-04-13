import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
from easygui import *
# 图片加载
img_monkey = io.imread('imgs/monkey.png')
img_woman = io.imread('imgs/woman.png')
# 尺寸相同化
img_monkey =cv2.resize(img_monkey, img_woman.shape[:2])

# 变换控制点的坐标列表
x_src = []
y_src = []
x_dst = []
y_dst = []
# ---------------选择控制点
# 眼睛
con = ccbox(msg='是否选择眼睛控制', title=' ', choices=(' yes ', ' no '), image=None)
if con == True:
    n = enterbox(msg=' 一只眼睛几个控制点 ', title=' ', default=' ', strip=True, image=None, root=None)
    n = int(n)
    plt.imshow(img_woman)
    plt.title('Choose Left Eye Control points')
    pos = plt.ginput(n)
    for i in range(0, n):
        x_src.append(round(pos[i][1]))
        y_src.append(round(pos[i][0]))
    plt.close()
    plt.imshow(img_monkey)
    plt.title('Choose Left Eye Control points')
    pos = plt.ginput(n)
    for i in range(0, n):
        x_dst.append(round(pos[i][1]))
        y_dst.append(round(pos[i][0]))
    plt.close()
    plt.imshow(img_woman)
    plt.title('Choose Right Eye Control points')
    pos = plt.ginput(n)
    for i in range(0, n):
        x_src.append(round(pos[i][1]))
        y_src.append(round(pos[i][0]))
    plt.close()
    plt.imshow(img_monkey)
    plt.title('Choose Right Eye Control points')
    pos = plt.ginput(n)
    for i in range(0, n):
        x_dst.append(round(pos[i][1]))
        y_dst.append(round(pos[i][0]))
    plt.close()
    # 鼻子
con2 = ccbox(msg='是否选择鼻子控制', title=' ', choices=(' yes ', ' no '), image=None)
if con2 == True:
    n = enterbox(msg=' 鼻子几个控制点 ', title=' ', default=' ', strip=True, image=None, root=None)
    n = int(n)
    plt.imshow(img_woman)
    plt.title('Choose Nose Control points')
    pos = plt.ginput(n)
    for i in range(0, n):
        x_src.append(round(pos[i][1]))
        y_src.append(round(pos[i][0]))
    plt.close()
    plt.imshow(img_monkey)
    plt.title('Choose Nose Control points')
    pos = plt.ginput(n)
    for i in range(0, n):
        x_dst.append(round(pos[i][1]))
        y_dst.append(round(pos[i][0]))
    plt.close()
    # 嘴巴
con3 = ccbox(msg='是否选择嘴巴控制', title=' ', choices=(' yes ', ' no '), image=None)
if con3 == True:
    n = enterbox(msg=' 嘴巴几个控制点 ', title=' ', default=' ', strip=True, image=None, root=None)
    n = int(n)
    plt.imshow(img_woman)
    plt.title('Choose Mouth Control points')
    pos = plt.ginput(n)
    for i in range(0, n):
        x_src.append(round(pos[i][1]))
        y_src.append(round(pos[i][0]))
    plt.close()
    plt.imshow(img_monkey)
    plt.title('Choose Mouth Control points')
    pos = plt.ginput(n)
    for i in range(0, n):
        x_dst.append(round(pos[i][1]))
        y_dst.append(round(pos[i][0]))
    plt.close()
# 额外设置四个角的控制点
[a1,b1]=img_woman.shape[:2]
x_src.extend([0,0,a1,a1])
y_src.extend([0,b1,0,b1])
x_dst.extend([0,0,a1,a1])
y_dst.extend([0,b1,0,b1])
# 得到控制点坐标
x_src2 = np.array(x_src)
x_dst2 = np.array(x_dst)
y_src2 = np.array(y_src)
y_dst2 = np.array(y_dst)
#----用于检查列表正确性
print(x_src2)
print(y_src2)
print(x_dst2)
print(y_dst2)