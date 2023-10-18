#!/usr/bin/env python3
"""
https://developers.google.com/sheets/api/quickstart/python
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib


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

import math
import sys
import cv2
import numpy as np
import pandas as pd
import imageio

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

import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


def get_credentials():
    # If modifying these scopes, delete the file token.json.
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds


def get_data(range_name):
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    SPREADSHEET_ID = "1f3XV-DQJXmObCLrgZcIM_DzttEhxCsKLHVntJAIrs4M"
    creds = get_credentials()
    service = build("sheets", "v4", credentials=creds)

    # Call the Sheets API
    sheet = service.spreadsheets()
    result = (
        sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=range_name).execute()
    )
    values = result.get("values", [])
    if not values:
        raise RuntimeError("No data found.")
    return values


def create_dataframe(data):
    data[0][0] = "Time"
    df = pd.DataFrame(data[1:], columns=data[0])
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M")
    df.set_index("Time", inplace=True)
    df = df.replace("[\£,]", "", regex=True).astype(float)
    return df


def setup_dataframes():
    fnb_data = create_dataframe(get_data(range_name="Sales Heat Map Data!A3:H20"))
    ck_data = create_dataframe(get_data(range_name="Sales Heat Map Data!A28:H45"))
    trApt_data = create_dataframe(get_data(range_name="Sales Heat Map Data!A57:H74"))
    trApt_data = create_dataframe(get_data(range_name="Sales Heat Map Data!A57:H74"))
    cube_data = create_dataframe(get_data(range_name="Sales Heat Map Data!A82:H99"))
    # ua_data = create_dataframe(get_data(range_name="Sales Heat Map Data!A107:H124"))
    events_data = create_dataframe(get_data(range_name="Sales Heat Map Data!A132:H149"))

    fnb_data.to_pickle("fnb_data.pkl")
    ck_data.to_pickle("ck_data.pkl")
    trApt_data.to_pickle("trApt_data.pkl")
    cube_data.to_pickle("cube_data.pkl")
    # ua_data.to_pickle("ua_data.pkl")
    events_data.to_pickle("events_data.pkl")


def mask_from_image(mask_image):
    mask_img = cv2.imread(mask_image)
    imgray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    _, thresh_binary = cv2.threshold(imgray, 10, 255, cv2.THRESH_BINARY)
    return thresh_binary


def printCount(array):
    from collections import Counter

    count = Counter()
    width, height = array.shape[:2]
    for w in range(width):
        for h in range(height):
            v = array[w, h]
            count[v] += 1
    print("Count: ", count)


# def color_map(input, dest, color_map):
#   # https://stackoverflow.com/questions/28825520/is-there-something-like-matlabs-colorbar-for-opencv

#   num_bar_w=30
#   color_bar_w=10
#   vline=10

#   width, height = input.shape[:2]
#   win_mat = np.full((width+num_bar_w+num_bar_w+vline, height,3), (255,255,255), np.uint8)

#   mmin = int(input.min())
#   mmax = input.max()
#   max_int = math.ceil(mmax)
#   input = input * (255.0/(mmax-mmin))
#   input = input.astype(np.uint8)

#   M = cv2.applyColorMap(input, color_map);
#   M.copyTo(win_mat(cv2.Rect( 0, 0, width, height)))

#   # Scale
#   num_window = np.full((num_bar_w, height,3), (255,255,255), np.uint8)

#   for i in range(max_int):
#       j=i*input.rows/max_int
#       cv2.putText(num_window, i, (5, num_window.rows-j-5),cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (0,0,0), 1 , 2 , False)

#   #color bar
#   color_bar = np.full((color_bar_w, height,3), (255,255,255), np.uint8)
#   cv::Mat cb;
#   for i in range(height):
#     for j in range(color_bar_w):
#       v = 255-255*i/height;
#       color_bar.at<cv::Vec3b>(i,j)=cv::Vec3b(v,v,v)

#   color_bar.convertTo(color_bar, CV_8UC3);
#   ccb = cv2.applyColorMap(color_bar, color_map);
#   num_window.copyTo(win_mat(cv::Rect(input.cols+vline+color_bar_w, 0, num_bar_w, input.rows)));
#   cb.copyTo(win_mat(cv::Rect(input.cols+vline, 0, color_bar_w, input.rows)));
#   dest=win_mat.clone();


#
# Set up Data
#
# setup_dataframes()
fnb_data = pd.read_pickle("fnb_data.pkl")
ck_data = pd.read_pickle("ck_data.pkl")
# trApt_data = pd.read_pickle("trApt_data.pkl")
cube_data = pd.read_pickle("cube_data.pkl")
# ua_data = pd.readpickle("ua_data.pkl")
# events_data = pd.read_pickle("events_data.pkl")

# Get min and max values
min_value = 0
max_value = max(
    fnb_data.to_numpy().max(), ck_data.to_numpy().max(), cube_data.to_numpy().max()
)

#
# Set up Image data
#
map_img = cv2.imread("/opt/heatmap/farm urban rooftop v1.png")
width, height = map_img.shape[:2]
mask_fnb = mask_from_image("/opt/heatmap/FnB.png")
mask_ck = mask_from_image("/opt/heatmap/CK.png")
mask_cube = mask_from_image("/opt/heatmap/Cube.png")

# Add data to heatmap
# Mask: zero:255 - 255 is ROI
# np.where(mask1, BLUE, map_img)

frames = []
for day in [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]:
    for i, fnb in enumerate(fnb_data[day].items()):
        timestamp = fnb[0]
        time = timestamp.strftime("%H:%M")
        heatmap_mask = np.full((width, height), 0, np.uint8)
        heatmap_mask[mask_fnb == 255] = fnb[1] if fnb[1] > 0 else 1
        heatmap_mask[mask_ck == 255] = (
            ck_data[day][timestamp] if ck_data[day][timestamp] > 0 else 1
        )
        heatmap_mask[mask_cube == 255] = (
            cube_data[day][timestamp] if cube_data[day][timestamp] > 0 else 1
        )
        # print(fdata, ck_data[day][timestamp], cube_data[day][timestamp])

        # Add in the min and max values to the colour map is consistent across runs
        heatmap_mask[0, 0] = min_value
        heatmap_mask[0, 1] = max_value

        # Create the heatmap image
        heatmap_img = cv2.applyColorMap(heatmap_mask, cv2.COLORMAP_HOT)

        # Normalise the heatmap_mask and make binary
        cv2.normalize(heatmap_mask, heatmap_mask, 0, 255, cv2.NORM_MINMAX)
        heatmap_mask[heatmap_mask > 0] = 255
        heatmap_mask_inv = cv2.bitwise_not(heatmap_mask)

        # Now black-out the area where we will put the heatmap
        bg = cv2.bitwise_and(map_img, map_img, mask=heatmap_mask_inv)

        # Take only region of heatmap from heatmap image.
        fg = cv2.bitwise_and(heatmap_img, heatmap_img, mask=heatmap_mask)

        merged = cv2.add(bg, fg)

        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (60, 50)
        fontScale = 1
        fontColor = (0, 0, 0)
        thickness = 3
        lineType = 2

        cv2.putText(
            merged,
            f"{day} {time}",
            position,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )
        frames.append(merged)
        # cv2.imshow("MERGED", merged)
        # cv2.waitKey(0)

imageio.mimsave("urbanoasis.gif", frames, fps=55)

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
