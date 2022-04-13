import matplotlib.pyplot as plt
from skimage import io, exposure,color
path='图片库\cell.tif'
img = io.imread(path)
rows,cols=img.shape[:2]

plt.subplot(121)
plt.title('origin image')
io.imshow(img)
plt.axis('off')

r2=img.max()-1
r1=img.min()+1 
s1=50
s2=200
i2=img

for i in range(rows):
    for j in range(cols):
        if (img[i,j]<=r1):
             i2[i,j]= img[i,j]*s1/(r1+1)
        elif  (img[i,j]>=r2):
             i2[i,j]= (img[i,j]-r2)*(255-s2)/(255-r2)+s2
        else :
            i2[i,j]= (img[i,j]-r1)*(s2-s1)/(r2-r1)+s1

plt.subplot(122)
plt.title('s1=50 s2=200 r1=min r2=max')
io.imshow(i2)
plt.axis('off')

plt.show()
