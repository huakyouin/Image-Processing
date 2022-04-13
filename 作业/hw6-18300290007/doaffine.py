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
    x_src=[65,52,74,77,70,49,65,79,139,73,142,159,111,178,178,179,213,213,0,0,258,258]
    y_src=[42,69,96,70,162,189,225,192,96,128,162,127,130,80,130,177,148,111,0,256,0,256]
    x_dst=[26,21,39,41,33,18,23,38,185,57,195,220,112,211,230,206,245,244,0,0,258,258]
    y_dst=[67,88,108,86,145,165,185,169,77,124,167,118,126,63,119,194,154,98,0,256,0,256]
    x_src2 = np.array(x_src)
    x_dst2 = np.array(x_dst)
    y_src2 = np.array(y_src)
    y_dst2 = np.array(y_dst)
    # 只看后向图
    plt.subplot(2,2,1)
    new_img_back = backward_transform(img_woman, x_dst2, y_dst2, x_src2, y_src2, 1)
    plt.imshow(new_img_back)
    plt.title('e=1')
    plt.subplot(2,2,2)
    new_img_back = backward_transform(img_woman, x_dst2, y_dst2, x_src2, y_src2, 2)
    plt.imshow(new_img_back)
    plt.title('e=2')
    plt.subplot(2,2,3)
    new_img_back = backward_transform(img_woman, x_dst2, y_dst2, x_src2, y_src2, 3)
    plt.imshow(new_img_back)
    plt.title('e=3')
    plt.subplot(2,2,4)
    new_img_back = backward_transform(img_woman, x_dst2, y_dst2, x_src2, y_src2, 4)
    plt.imshow(new_img_back)
    plt.title('e=4')
    plt.show()
