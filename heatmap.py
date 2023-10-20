#!/usr/bin/env python3
"""

To display on a Mac: qlmanage -p urbanoasis.gif

Data: https://docs.google.com/spreadsheets/d/1f3XV-DQJXmObCLrgZcIM_DzttEhxCsKLHVntJAIrs4M/edit#gid=1221641764


Creating Masks with GIMP:
* Use Path Tool to select area
* ⌘-click on start point to close path
* Select -> Invert to select the background
* ⌘-X to cut the background
* Select -> Invert to select the ROI
* Use the Bucket Fill tool to fill the area with Blue
* Export as PNG

# Copying bits of images over using masks
https://stackoverflow.com/questions/41572887/equivalent-of-copyto-in-python-opencv-bindings



https://dev.to/kuba_szw/what-is-the-most-interesting-place-in-the-backyard-make-yourself-a-heatmap-2k7b

https://gist.github.com/Tushar-N/58e9432db69ced0ac933b8e662bc2da2

https://stackoverflow.com/questions/46020894/superimpose-heatmap-on-a-base-image-opencv-python

Applying just to areas
https://medium.com/omdena/visualizing-pathologies-in-ultrasound-image-using-opencv-and-streamlit-73b6f4b67c37


Matplotlib, Scipy
https://github.com/LinShanify/HeatMap

https://stackoverflow.com/questions/67117074/how-to-add-a-data-driven-location-based-heatmap-to-some-image


"""

import math
import sys
import cv2
import numpy as np
import pandas as pd
import imageio

import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


def get_credentials():
    """
    https://developers.google.com/sheets/api/quickstart/python
    pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
    """
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


def mask_center(mask):
    countors, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(countors) == 0:
        raise RuntimeError("No countors found")
    if len(countors) > 1:
        raise RuntimeError("More than one countor found")
    mask_countor = countors[0]
    x, y, w, h = cv2.boundingRect(mask_countor)
    return (int(x + w / 2), int(y + h / 2))


def mask_add_text(text, mask_center, image):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255)
    thickness = 2
    line_type = 2
    bottom_left_origin = False

    box_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)

    px = int(mask_center[0] - (box_size[0] / 2))
    py = int(mask_center[1])
    position = (px, py)

    cv2.putText(
        image,
        text,
        position,
        font_face,
        font_scale,
        font_color,
        thickness,
        line_type,
        bottom_left_origin,
    )


def printCount(array):
    from collections import Counter

    count = Counter()
    width, height = array.shape[:2]
    for w in range(width):
        for h in range(height):
            v = array[w, h]
            if type(v) == np.ndarray:
                v = str(v)
            count[v] += 1
    print("Count: ", count)


def add_colourbar(input, imax, colour_map=cv2.COLORMAP_JET):
    # if input is None:
    #     width = 400
    #     height = 300
    #     input = np.random.randint(0, 255, (height, width, 3), np.uint8)

    height, width, _ = input.shape
    color_bar_w = 20
    num_bar_w = 50
    spacer = 10

    win_mat = np.full(
        (height, width + num_bar_w + num_bar_w + spacer, 3),
        (255, 255, 255),
        np.uint8,
    )
    # if imin is None:
    #     heat_map = cv2.applyColorMap(input, colour_map)
    #     imin = int(heat_map.min())
    #     imax = math.ceil(heat_map.max())

    # # Normalise Data: https://stackoverflow.com/questions/46689428/convert-np-array-of-type-float64-to-type-uint8-scaling-values
    # info = np.iinfo(data.dtype)  # Get the information of the incoming image type
    # data = data.astype(np.float64) / info.max  # normalize the data to 0 - 1
    # data = 255 * data  # Now scale by 255

    # Copy heatmap into the blank win_mat in the rect area - so rect area is mask
    win_mat[0:height, 0:width] = input

    # color bar
    colour_bar = np.full((height, color_bar_w, 3), (255, 255, 255), np.uint8)
    for i in range(height):
        for j in range(color_bar_w):
            # v = 255 - 255 * i / height
            v = 255 * i / height
            colour_bar[i, j] = (v, v, v)
    colour_bar = cv2.applyColorMap(colour_bar, colour_map)

    # Scale
    num_bar = np.full((height, num_bar_w, 3), (255, 255, 255), np.uint8)
    offset = 10
    font_scale = 0.4
    font_color = (0, 0, 0)
    thickness = 1
    line_type = 2
    for i in range(0, imax, imax // 10):
        j = i * (height - offset) / imax
        cv2.putText(
            num_bar,
            str(i),
            (offset, int(j + offset)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_color,
            thickness,
            line_type,
            False,
        )

    # Copy in images
    win_mat[0:height, 0:width] = input
    istart = width + spacer
    win_mat[0:height, istart : istart + color_bar_w] = colour_bar
    istart = istart + color_bar_w + spacer
    win_mat[0:height, istart : istart + num_bar_w] = num_bar
    return win_mat


COLOUR_MAP = cv2.COLORMAP_COOL
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
# Get centers of mask
mask_fnb_center = mask_center(mask_fnb)
mask_ck_center = mask_center(mask_ck)
mask_cube_center = mask_center(mask_cube)


# Add data to heatmap
# Mask: zero:255 - 255 is ROI
# np.where(mask1, BLUE, map_img)

frames = []
previous = None
fnb_value_prev = None
ck_value_prev = None
cube_value_prev = None
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
        print(f"Day: {day} {i} {fnb}")
        timestamp = fnb[0]
        time = timestamp.strftime("%H:%M")
        heatmap_mask = np.full((width, height), 0.0, np.double)
        fnb_value = fnb[1]
        # fnb_value = 100
        heatmap_mask[mask_fnb == 255] = fnb_value
        ck_value = ck_data[day][timestamp]
        # ck_value = 200
        heatmap_mask[mask_ck == 255] = ck_value
        cube_value = cube_data[day][timestamp]
        # cube_value = 300
        heatmap_mask[mask_cube == 255] = cube_value

        # Add in the max value so the colour map is consistent across runs
        heatmap_mask[0, 0] = max_value
        # print(f"FNB: {fnb_value} CK: {ck_value} Cube: {cube_value}")
        # printCount(heatmap_mask)

        # Create the heatmap image
        cv2.normalize(heatmap_mask, heatmap_mask, 0, 255, cv2.NORM_MINMAX)
        heatmap_mask = heatmap_mask.astype(np.uint8)
        # printCount(heatmap_mask)
        heatmap_img = cv2.applyColorMap(heatmap_mask, COLOUR_MAP)

        # Normalise the heatmap_mask and make binary
        # cv2.normalize(heatmap_mask, heatmap_mask, 0, 255, cv2.NORM_MINMAX)
        heatmap_mask[heatmap_mask > 0] = 255
        heatmap_mask_inv = cv2.bitwise_not(heatmap_mask)

        # Now black-out the area where we will put the heatmap
        bg = cv2.bitwise_and(map_img, map_img, mask=heatmap_mask_inv)
        # Take only region of heatmap from heatmap image.
        fg = cv2.bitwise_and(heatmap_img, heatmap_img, mask=heatmap_mask)
        merged = cv2.add(bg, fg)

        # Add colour bar
        merged = add_colourbar(merged, int(max_value), COLOUR_MAP)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        position = (60, 50)
        font_scale = 1
        font_color = (0, 0, 0)
        thickness = 3
        line_type = 2
        cv2.putText(
            merged,
            f"{day:9} {time}",
            position,
            font_face,
            font_scale,
            font_color,
            thickness,
            line_type,
        )

        merged_notext = merged.copy()
        if fnb_value > 0:
            mask_add_text(str(int(fnb_value)), mask_fnb_center, merged)
            if fnb_value == fnb_value_prev:
                mask_add_text(str(int(fnb_value)), mask_fnb_center, merged_notext)
        fnb_value_prev = fnb_value
        if ck_value > 0:
            mask_add_text(str(int(ck_value)), mask_ck_center, merged)
            if ck_value == ck_value_prev:
                mask_add_text(str(int(fnb_value)), mask_ck_center, merged_notext)
        ck_value_prev = ck_value
        if cube_value > 0:
            mask_add_text(str(int(cube_value)), mask_cube_center, merged)
            if cube_value == cube_value_prev:
                mask_add_text(str(int(fnb_value)), mask_cube_center, merged_notext)
        cube_value_prev = cube_value

        if previous is not None:
            for ifade in range(10):
                alpha = ifade / 10
                beta = 1 - alpha
                if ifade > 5:
                    merged = merged_notext
                merged_ = cv2.addWeighted(merged, alpha, previous, beta, 0)
                # cv2.imshow("MERGED2", merged_)
                # cv2.waitKey(0)
                frames.append(merged_)
        else:
            frames.append(merged)
        previous = merged
        # cv2.imshow("MERGED1", merged)
        # cv2.waitKey(0)

imageio.mimsave("urbanoasis.gif", frames, fps=10)
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
