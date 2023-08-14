import cv2
import matplotlib.pyplot as plt
import numpy as np
A = cv2.imread(r'C:/Users/91944/Downloads/fake-currency-detection/Real.jpg')
P = cv2.imread(r'C:\Users\91944\Downloads\fake-currency-detection\fake.jpg')
plt. Subplot(531)
plt.imshow(A)
a = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
p = cv2.cvtColor(P, cv2.COLOR_BGR2GRAY)
plt.subplot(532)
plt.imshow(a)
a2tr = a[330:1200, 1016:1927]
plt.subplot(533)
plt.imshow(a2tr)
b2tr = p[170:1040, 716:1627]
plt.subplot(534)
plt.imshow(b2tr)
print(a.shape)
a2_str = a[5:1100, 2080:2151]
plt.subplot(535)
plt.imshow(a2_str)
print(p.shape)
p2_str = p[5:1100, 1666:1729]
plt.imshow(p2_str)
hsvImageReal = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
hsvImageFake = cv2.cvtColor(P, cv2.COLOR_BGR2HSV)
plt.subplot(536)
plt.imshow(hsvImageReal)
plt.subplot(537)
plt.imshow(hsvImageFake)
croppedImageReal = hsvImageReal[5:1100, 2080:2151]
plt.subplot(538)
plt.imshow(croppedImageReal)
croppedImageFake = hsvImageFake[5:1100, 1666:1729]
plt.subplot(539)
plt.imshow(croppedImageFake)
satThresh = 0.3
valThresh = 0.9
g = croppedImageReal[:,:,1]>satThresh
h = croppedImageReal[:,:,2] < valThresh
g1 = croppedImageFake[:,:,1]>satThresh
h1 = croppedImageFake[:,:,2] < valThresh
BWImageReal = g&h
BWImageFake = g1&h1
def bwareaopen(img, min_size, connectivity=8):
   
    # Find all connected components (called here "labels")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=connectivity)
   
    # check size of all connected components (area in pixels)
    for i in range(num_labels):
        label_size = stats[i, cv2.CC_STAT_AREA]
       
        # remove connected components smaller than min_size
        if label_size < min_size:
            img[labels == i] = 0
           
    return img
binr = cv2.threshold(a2_str, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
 
# define the kernel
kernel = np.ones((3, 3), np.uint8)
 
# invert the image
invert = cv2.bitwise_not(binr)
 
# use morph gradient
BWImageCloseReal = cv2.morphologyEx(invert, cv2.MORPH_GRADIENT, kernel)
binr2 = cv2.threshold(p2_str, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
 
# define the kernel
kernel2 = np.ones((3, 3), np.uint8)
 
# invert the image
invert2 = cv2.bitwise_not(binr2)
 
# use morph gradient
BWImageCloseFake = cv2.morphologyEx(invert2, cv2.MORPH_GRADIENT, kernel2)
areaopenReal = bwareaopen(BWImageCloseReal, 15);
areaopenFake = bwareaopen(BWImageCloseFake, 15);
bw = areaopenReal
 
labels = np.zeros(bw.shape)
countReal = cv2.connectedComponentsWithStats(bw, labels,8);
bw2 = areaopenFake
 
labels2 = np.zeros(bw2.shape)
countFake = cv2.connectedComponentsWithStats(bw2, labels2,8);
def corr2(A, B):
   
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
 
    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)
    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))
co=corr2 (a2tr, b2tr)
 
if (co.any()>=0.5):
    print ('correlevance of transparent gandhi > 0.5')
    if (countReal[0] == countFake[0] ):
        print ('currency is legitimate')
    else:
        print ('green strip is fake')
else:
    print ('correlevance of transparent gandhi < 0.5')
    print ('currency is fake')
