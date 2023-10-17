#!/usr/bin/env python3
"""
https://dev.to/kuba_szw/what-is-the-most-interesting-place-in-the-backyard-make-yourself-a-heatmap-2k7b

https://gist.github.com/Tushar-N/58e9432db69ced0ac933b8e662bc2da2

https://stackoverflow.com/questions/46020894/superimpose-heatmap-on-a-base-image-opencv-python

Applying just to areas
https://medium.com/omdena/visualizing-pathologies-in-ultrasound-image-using-opencv-and-streamlit-73b6f4b67c37


Matplotlib, Scipy
https://github.com/LinShanify/HeatMap

https://stackoverflow.com/questions/67117074/how-to-add-a-data-driven-location-based-heatmap-to-some-image


Data: https://docs.google.com/spreadsheets/d/1f3XV-DQJXmObCLrgZcIM_DzttEhxCsKLHVntJAIrs4M/edit#gid=1221641764
* need to have a mask array for each area
* create an heatmap the size of the image initialised with zeros
* for each mask apply a value within the mask based on the data for that area to the heatmap
* merge the heatmap with the image


GIMP
Use Path Tool to select area
Apple-click on start point to close path
Select -> Invert to select the background
Apple-X to cut the background
Select -> Invert to select the ROI
Use the Bucket Fill tool to fill the area with Blue
Export as PNG

Selec
Ctrl-C to copy
Ctrl-V to paste as new layer



"""

import sys
import cv2
import numpy as np

# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon
# polygon = Polygon(bc)
# point = Point(h, w)
# if polygon.contains(point):


GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def mask_from_image(mask_image):
    mask_img = cv2.imread(mask_image)
    # width, height = mask_img.shape[:2]
    # print(f"GOT MASK {width}x{height}")
    imgray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    _, thresh_binary = cv2.threshold(imgray, 10, 255, cv2.THRESH_BINARY)
    return thresh_binary

    # contours, _ = cv2.findContours(
    #     thresh_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    # )
    # if len(contours) == 0:
    #     raise RuntimeError("No contours found")
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # bounding_contour = contours[1]
    # # cv2.drawContours(map_img, [bounding_contour], 0, (255, 0, 0), 3)
    # # cv2.imshow("THRESH_BINARY", map_img)
    # # cv2.waitKey(0)
    # # print("BOUNDING ", bounding_contour)
    # # floodFill might be quicker?
    # mask = np.full((height, width), False, dtype=bool)
    # mask_fill = True
    # for h in range(height):
    #     for w in range(width):
    #         if cv2.pointPolygonTest(bounding_contour, (w, h), False) >= 0:
    #         mask[h, w] = mask_fill
    # return mask


map_img = cv2.imread("/opt/heatmap/farm urban rooftop v1.png")
width, height = map_img.shape[:2]
mask_image1 = "/opt/heatmap/Area1.png"
mask_image2 = "/opt/heatmap/Area2.png"
mask_image3 = "/opt/heatmap/Area3.png"

fAndBsales = {
    "Monday": {
        "08:00": 50,
        "10:00": 150,
        "09:00": 250,
        "11:00": 1000,
        "12:00": 1000,
        "13:00": 1000,
        "14:00": 300,
        "15:00": 145,
        "16:00": 500,
        "17:00": 1000,
        "18:00": 2000,
        "19:00": 2450,
        "20:00": 900,
        "21:00": 200,
        "22:00": 100,
        "23:00": 50,
        "00:00": 50,
    }
}


# print(f"GOT {width}x{height}")


# for' ij in np.ndindex(mask1.shape[:2]):
#     print(ij, mask1[ij])# blu'e = np.full((height, width, 3), WHITE, np.uint8)

# blue[mask > 0] = BLUE
# map_img[mask1 == 0] = BLUE
# map_img[mask2 == 0] = RED
# map_img[mask3 == 0] = GREEN
# np.where(mask1, BLUE, map_img)


# Get min/max of data - needed to put all data on the same scale
def dataMinMax(data):
    min = 256
    max = -1
    for day, byhour in data.items():
        for k, v in byhour.items():
            if v < min:
                min = v
            if v > max:
                max = v
    return min, max


def printCount(array):
    from collections import Counter

    count = Counter()
    width, height = array.shape[:2]
    for w in range(width):
        for h in range(height):
            v = array[w, h]
            count[v] += 1
    print("Count: ", count)


# heatmap_image = np.zeros((width, height, 1), np.uint8)
# heatmap_image = np.zeros((width, height), np.uint8)
mask1 = mask_from_image(mask_image1)
mask2 = mask_from_image(mask_image2)
mask3 = mask_from_image(mask_image3)

# Add data to heatmap
# Mask: zero:255 - 255 is ROI
heatmap_mask = np.full((width, height), 0, np.uint8)
heatmap_mask[mask1 == 255] = 10
heatmap_mask[mask2 == 255] = 100
heatmap_mask[mask3 == 255] = 200

# # Create alpha channel where uninteresting regions are transparent (0 in alpha is transparent)
# alphaH = np.full((width, height), 0, np.uint8)
# alphaH[heatmap_image != 255] = 255
# cv2.imshow("THRESH_BINARY1", alphaH)
# cv2.waitKey(0)
# NB MIGHT NEED TO INVERT THIS

# Colour the heatmap
heatmap_img = cv2.applyColorMap(heatmap_mask, cv2.COLORMAP_JET)

# Make heatmap_mask binary
heatmap_mask[heatmap_mask > 0] = 255
heatmap_mask_inv = cv2.bitwise_not(heatmap_mask)

# Now black-out the area where we will put the heatmap
bg = cv2.bitwise_and(map_img, map_img, mask=heatmap_mask_inv)
cv2.imshow("THRESH_BINARY1", bg)
cv2.waitKey(0)


# Take only region of heatmap from heatmap image.
fg = cv2.bitwise_and(heatmap_img, heatmap_img, mask=heatmap_mask)

cv2.imshow("THRESH_BINARY1", fg)
cv2.waitKey(0)

merged = cv2.add(bg, fg)
cv2.imshow("THRESH_BINARY1", merged)
cv2.waitKey(0)
sys.exit()

# Add alpha channel to heatmap: https://stackoverflow.com/questions/32290096/python-opencv-add-alpha-channel-to-rgb-image
# heatmap_img = np.dstack((heatmap_img, alphaH))

# Add alpha channel to map_img
# alphaI = np.full((width, height), 255, np.uint8)
# map_img = np.dstack((map_img, alphaI))


# heatmap_img = cv2.multiply(alphaH, heatmap_img)
# map_img = cv2.multiply(1 - alphaH, map_img)
# super_imposed_img = cv2.add(map_img, heatmap_img)

# super_imposed_img = cv2.addWeighted(heatmap_img, 1, map_img, 1, 0)

cv2.imshow("FOO", super_imposed_img)
cv2.waitKey(0)
sys.exit()

# imgray_i = cv2.bitwise_not(imgray)


heatmap_image = np.zeros((height, width, 1), np.uint8)


# Check values of ksize and 0
ksize = 999999
hmap = cv2.GaussianBlur(heatmap, (k_size, k_size), 0)
# hmap = hmap/hmap.max()
# hmap = (hmap*255).astype(np.uint8)
heatmap_img = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
