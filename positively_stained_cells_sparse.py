from __future__ import print_function
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed

import cv2 as cv
import imutils
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io

#1 9 5 for DAB_seg.png & Figure_HE.png
#1 1 1 for DAB3_seg.png

#input the DAB image obtained from ColorUnmixing.py
user_input = input ("Enter the file name of the positively stained cells image (output from ColorUnmixing.py): ")
img = cv.imread(user_input)
#white objects on black background 
img = cv.bitwise_not(img)
shifted = cv.pyrMeanShiftFiltering(img, 21, 51)
gray = cv.cvtColor(shifted, cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

userInput1 = int(input ("Enter the no. of iterations for [open]: "))
userInput2 = int(input ("Enter the no. of iterations for [dilate]: "))
userInput3 = int(input ("Enter the min. distance (in pixels) between peaks (1-20): "))

kernel = np.ones((3,3),np.uint8)
thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = userInput1)
thresh = cv.dilate(thresh, None, iterations=userInput2)
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=userInput3,labels=thresh)
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("No. of positively stained cells: {}".format(len(np.unique(labels)) - 1))

userInput = input ("Enter the file name of the all-cells image (output from ColorUnmixing.py) for accuracy check: ")
img_rgb = io.imread(userInput)
#img_rgb = cv.imread(userInput,cv.COLOR_BGR2RGB)
for label in np.unique(labels):
    if label == 0:
        continue
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv.contourArea)
    ((x, y), r) = cv.minEnclosingCircle(c)
    cv.circle(img_rgb, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv.putText(img_rgb, "#{}".format(label), (int(x) - 10, int(y)), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1) 

fig = plt.figure(frameon=False)
plt.rcParams['figure.figsize'] = 10, 10
plt.imshow(img_rgb)
plt.axis('off')
long_title = 'Settings : no. of iterations =  {} & {} & min. distance = {}'
plt.title(long_title.format(userInput1, userInput2, userInput3))
plt.show()
