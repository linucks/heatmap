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
    width, height = mask_img.shape[:2]
    # print(f"GOT MASK {width}x{height}")
    imgray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    # return imgray
    _, thresh_binary = cv2.threshold(imgray, 10, 255, cv2.THRESH_BINARY_INV)
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


# print(f"GOT {width}x{height}")

mask1 = mask_from_image(mask_image1)
mask2 = mask_from_image(mask_image2)
mask3 = mask_from_image(mask_image3)

# for ij in np.ndindex(mask1.shape[:2]):
#     print(ij, mask1[ij])

# blue = np.full((height, width, 3), WHITE, np.uint8)
# blue[mask > 0] = BLUE
map_img[mask1 == 0] = BLUE
map_img[mask2 == 0] = BLUE
map_img[mask3 == 0] = BLUE
# np.where(mask1, BLUE, map_img)


def colour_img(img, mask, colour):
    width, height = img.shape[:2]
    for h in range(height):
        for w in range(width):
            if mask[h, w]:
                img[h, w] = colour


# colour_img(map_img, mask1, BLUE)
# colour_img(map_img, mask2, RED)

cv2.imshow("THRESH_BINARY", map_img)
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
