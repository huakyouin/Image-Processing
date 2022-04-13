import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
from easygui import *

# 得到控制点的变换矩阵T
def get_matrix_T(x_src, y_src, x_dst, y_dst):
    # 获取控制点个数
    num_of_point = len(x_src)
    # 初始化T
    T = np.zeros((num_of_point, 3, 3))
    # 得到各个控制点的变换矩阵，实质是一个平移矩阵
    for i in range(num_of_point):
        T[i] = np.array([[1, 0, x_dst[i] - x_src[i]], [0, 1, y_dst[i] - y_src[i]], [0, 0, 1]])
    return T


# 获得距离
def get_dist(h, w, x, y):
    dist = np.sqrt((h - x) ** 2 + (w - y) ** 2)
    # 防止距离为0影响后续计算
    eps = 1e-8
    if dist <= eps:
        return eps
    else:
        return dist


# 得到各点到各个控制点的权重值
def get_weight_by_dist(h, w, X, Y, e):
    num_of_point = X.shape[0]
    # 权重矩阵
    W = np.zeros((num_of_point, 1))
    for i in range(num_of_point):
        x, y = X[i], Y[i]
        W[i] = 1 / (get_dist(h, w, x, y)**e)
    W /= np.sum(W)
    return W


# 得到各点仿射后的位置
def get_mean_transform_point(h, w, T, X, Y, e):
    num = T.shape[0]
    W = get_weight_by_dist(h, w, X, Y, e)

    D = np.zeros((num, 2))
    # 计算出受各个控制点的变换矩阵T所应该映射到的位置
    for i in range(num):
        scr_point = np.array([h, w, 1]).transpose()
        dst_point = np.matmul(T[i], scr_point)
        dst_point = dst_point.transpose()
        # get new_h and new_w
        nh, nw, _ = dst_point
        D[i] = np.array([nh, nw])

    mean_d = np.zeros(2)
    # 根据权重计算在所有控制点作用下的位置
    for i in range(num):
        D[i] = W[i] * D[i]
        mean_d += D[i]

    return mean_d

# 前向变换
def forward_transform(img, x_src, y_src, x_dst, y_dst, e):

    new_img = np.zeros_like(img)
    H, W, C = new_img.shape

    # 对每个像素点标志是否已经填过
    is_paint = np.zeros((H,W),dtype='uint8')

    T = get_matrix_T(x_src, y_src, x_dst, y_dst)

    # 逐点变换
    for h in range(H):
        for w in range(W):
            nh, nw = get_mean_transform_point(h, w, T, x_src, y_src, e)
            nh, nw = int(np.round(nh)), int(np.round(nw))
            if nh < 0 or nw < 0 or nh >= H or nw >= W:
                continue
            new_img[nh, nw] = img[h, w]
            # 已经填过的点标记一下
            is_paint[nh, nw] = 1

    # 对没有填的区域进行修复
    inpaint_mask = 1 - is_paint
    new_img = cv2.inpaint(new_img, inpaint_mask, 13, cv2.INPAINT_NS)

    # blur
    new_img = cv2.blur(new_img, (5, 5))

    # median blur for denoise
    new_img = cv2.medianBlur(new_img, 7)
    return new_img


# 双线性插值
def biInterpolate(x, i, j):
    rows, cols, C = x.shape
    up = int(np.floor(i))
    down = int(np.ceil(i))
    left = int(np.floor(j))
    right = int(np.ceil(j))
    if up < 0 or left < 0 or down >= rows or right >= cols:
        return 0
    u, v = i - up, j - left
    y = u * v * x[up, left] + u * (1 - v) * x[up, right] + (1 - u) * v * x[down, left] + (1 - u) * (1 - v) * x[
        down, right]
    return y


# 后向变换
def backward_transform(img, x_src, y_src, x_dst, y_dst, e):
    new_img = np.zeros_like(img)
    H, W, C = new_img.shape
    T = get_matrix_T(x_src, y_src, x_dst, y_dst)

    for h in range(H):
        for w in range(W):
            nh, nw = get_mean_transform_point(h, w, T, x_src, y_src, e)

            img_gray_by_interpotate = biInterpolate(img, nh, nw)

            new_img[h, w] = img_gray_by_interpotate

    # 模糊化
    new_img = cv2.blur(new_img, (3, 3))
    return new_img


if __name__ == '__main__':
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
    # 参数e(权重公式中的)
    e = 2
    # 前向
    new_img_forward = forward_transform(img_woman, x_src2, y_src2, x_dst2, y_dst2, e)
    # 后向
    new_img_back = backward_transform(img_woman, x_dst2, y_dst2, x_src2, y_src2, e)
    savepath_forward = 'results/forward_result—6112032.png'
    savepath_back = 'results/back_result—6112032.png'
    io.imsave(savepath_forward, new_img_forward)
    io.imsave(savepath_back, new_img_back)