import shutil
import cv2
import numpy as np
import time
import os
import sys
import operator
import glob
import random
import json
import base64

import tkinter
import tkinter.filedialog

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", default="")
args = parser.parse_args()

CONTOUR_MIN_AREA = 250
CONTOUR_MAX_AREA = 25000

if args.input_path == "":
    root = tkinter.Tk()
    root.withdraw()
    input_path = os.path.abspath(tkinter.filedialog.askdirectory(title = "Please select the input image folder",
                                                                 initialdir = os.getcwd()))
else:
    input_path = os.path.abspath(args.input_path)

# check to see if directory contains child pngs somewhere
print(os.path.join(input_path, "**/*"))
childimages = glob.glob(os.path.join(input_path, "**/*"), recursive = True)
childimages = list(filter(lambda x: os.path.splitext(os.path.split(x)[1])[1] in [".png", ".jpg"],
                    childimages))
childimages = list(filter(lambda x: "segmented" not in x, childimages))
childimages = list(filter(lambda x: "proofed" not in x, childimages))
if len(childimages) == 0:
    print("Expected path to contain input images! Quitting.")
    sys.exit(1)

input_path = os.path.split(childimages[0])[0]
output_path = os.path.join(input_path, "proofed")

print("Input path:")
print(input_path)
print("Output path:")
print(output_path)

os.makedirs(output_path, exist_ok=True)

candidateSegment = None
anColorsSaved = [[60, 60, 0], [0, 60, 60], [0, 0, 120]]
anColors = [[128, 128, 0], [0, 128, 128], [180, 20, 180]]
uiColors = [[128, 128, 0], [0, 128, 128], [180, 20, 180]]

current_draw_option_id = 0
draw_option_settings = [
    { "frame": True, "accept": True, "reject": True },
    { "frame": False, "accept": True, "reject": True },
    { "frame": True, "accept": False, "reject": False },
    { "frame": True, "accept": True, "reject": False },
]
draw_option = draw_option_settings[current_draw_option_id]

def cycleDrawOptions():
    global current_draw_option_id
    global draw_option
    current_draw_option_id += 1
    current_draw_option_id %= len(draw_option_settings)
    draw_option = draw_option_settings[current_draw_option_id]

def safedel(fname):
    if os.path.exists(fname):
        os.unlink(fname)

textColor = (150, 80, 60)

def overlaySegments(colorframe):
    if draw_option["frame"]:
        overlayed = colorframe.copy()
    else:
        overlayed = np.zeros_like(colorframe)

    if draw_option["accept"]:
        for sid in [0, 1]:
            overlayed[acceptPlanes[:,:,sid] != 0, 0] += anColorsSaved[sid][0]
            overlayed[acceptPlanes[:,:,sid] != 0, 1] += anColorsSaved[sid][1]
            overlayed[acceptPlanes[:,:,sid] != 0, 2] += anColorsSaved[sid][2]

    if draw_option["reject"]:
        overlayed[excludePlane != 0, 0] = 0     # B
        overlayed[excludePlane != 0, 1] = 0     # G
        overlayed[excludePlane != 0, 2] = 255   # R

    return overlayed

def showAll(colorframe, frameinfo):
    cv2.imshow("viewer", frameWithMarkup(overlaySegments(colorframe), frameinfo))

def screenshot(colorframe, frameinfo):
    import time
    fname = f"screenshot-{int(time.time())}.png"
    cv2.imwrite(fname, frameWithMarkup(overlaySegments(colorframe), frameinfo))

objectId = 2

def mouseEvent(event, x, y, flags, param):
    global dirtyImage
    global acceptPlanes
    global excludePlane
    if (event == cv2.EVENT_LBUTTONDOWN):
        if objectId in [0, 1]:
            # inpaint
            colorized = np.zeros((acceptPlanes.shape[0], acceptPlanes.shape[1], 3), dtype=np.uint8)
            colorized[acceptPlanes[:,:,0] != 0, 0] = 255
            colorized[acceptPlanes[:,:,1] != 0, 2] = 255
            colors = [[255, 0, 0], [0, 0, 255]]
            colorized = cv2.circle(colorized, (x,y), 3, colors[objectId], -1)
            acceptPlanes[:,:,0] = colorized[:,:,0]
            acceptPlanes[:,:,1] = colorized[:,:,2]
        if objectId == 2:
            # erasure
            colorized = np.zeros((excludePlane.shape[0], excludePlane.shape[1], 3), dtype=np.uint8)
            colorized[excludePlane != 0, :] = 255
            colorized = cv2.circle(colorized, (x,y), 3, (255, 255, 255), -1)
            excludePlane = colorized[:,:,0]
            acceptPlanes[excludePlane != 0, :] = 0
        dirtyImage = True

def frameWithMarkup(colorframe, frameinfo):
    colorframe = cv2.circle(colorframe, (30,30), 12,
                                        [255, 255, 255],
                                        -1)
    colorframe = cv2.circle(colorframe, (30,30), 9,
                                        [int(anColorsSaved[objectId][0]*0.9),
                                         int(anColorsSaved[objectId][1]*0.9),
                                         int(anColorsSaved[objectId][2]*0.9)],
                                        -1)

    labeledBarXWidth = colorframe.shape[1] * 2/3
    labeledBarXBase = colorframe.shape[1] * 1/6
    labeledBarYBase = colorframe.shape[0] - 60

    # bar at the bottom that shows which frames have been labeled
    for phase in ["chaff", "labeled", "proofed", "reticule"]:
        for i in frameDirectory.keys():
            f = frameDirectory[i]["framenum"]
            proofed = frameDirectory[i]["already_proofed"]
            labeled = frameDirectory[i]["already_labeled"]

            if phase == "chaff" and not proofed and not labeled:
                color = uiColors[1]
                height = 3
                colorframe = cv2.line(colorframe,
                                (int(f/maxFrame * labeledBarXWidth + labeledBarXBase), int(labeledBarYBase-height)),
                                (int(f/maxFrame * labeledBarXWidth + labeledBarXBase), int(labeledBarYBase+height)),
                                color, 3)
            if phase == "labeled" and labeled and not proofed:
                color = uiColors[0]
                height = 7
                colorframe = cv2.line(colorframe,
                                (int(f/maxFrame * labeledBarXWidth + labeledBarXBase), int(labeledBarYBase-height)),
                                (int(f/maxFrame * labeledBarXWidth + labeledBarXBase), int(labeledBarYBase+height)),
                                color, 3)
            if phase == "proofed" and proofed:
                color = uiColors[2]
                height = 10
                colorframe = cv2.line(colorframe,
                                (int(f/maxFrame * labeledBarXWidth + labeledBarXBase), int(labeledBarYBase-height)),
                                (int(f/maxFrame * labeledBarXWidth + labeledBarXBase), int(labeledBarYBase+height)),
                                color, 3)
            if phase == "reticule" and f == frameinfo["framenum"]:
                # Display a reticule to indicate current frame
                color = uiColors[1]
                if labeled:
                    color = uiColors[0]
                if proofed:
                    color = uiColors[2]
                colorframe = cv2.line(colorframe,
                                    (int(f/maxFrame * labeledBarXWidth + labeledBarXBase), int(labeledBarYBase-15)),
                                    (int(f/maxFrame * labeledBarXWidth + labeledBarXBase), int(labeledBarYBase+15)),
                                    color, 5)

    curframe = frameinfo["framenum"]
    percent = int(curframe / maxFrame * 100)

    status_str = f"Frame: {curframe} / {maxFrame} [{percent:02d}%].  Proofed {numProofed} / {numLabeled}"
    colorframe = cv2.putText(colorframe,
                             status_str,
                             (70, 40),
                             cv2.FONT_HERSHEY_PLAIN, 1.5, (80, 80, 120), 2)
    return colorframe

def acceptRightSizeContours():
    global saved_masks
    global acceptPlanes
    global excludePlane

    for sid in range(len(saved_masks)):
        seg = saved_masks[sid]
        contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        areas = list(map(cv2.contourArea, contours))
        for cidx in range(len(contours)):
            print(f"Cid: {cidx} - area {areas[cidx]}")
            if areas[cidx] < CONTOUR_MIN_AREA or areas[cidx] > CONTOUR_MAX_AREA:
                print(f"***** ContourId: {cidx} is too big/small [area: {areas[cidx]}]")
                colorized = np.zeros((excludePlane.shape[0], excludePlane.shape[1], 3), dtype=np.uint8)
                colorized = cv2.drawContours(colorized, [contours[cidx]], -1, (255, 255, 255), -1)
                excludePlane = colorized[:,:,0]
            else:
                print(f"Keeping contour {cidx} [area: {areas[cidx]}]")
                colorized = np.zeros((acceptPlanes.shape[0], acceptPlanes.shape[1], 3), np.uint8)
                colorized[acceptPlanes[:,:,sid] != 0] = 255
                colorized = cv2.drawContours(colorized, [contours[cidx]], -1, (255, 255, 255), -1)
                acceptPlanes[:,:,sid] = colorized[:,:,0]

def excludeMasksThatOverlap():
    global acceptPlanes
    global excludePlane

    dilate_kernel = np.ones(3*3).reshape(3,3)
    canvas = np.zeros((acceptPlanes.shape[0], acceptPlanes.shape[1], 3))
    for sid in [0, 1]:
        seg = acceptPlanes[:, :, sid]
        canvas[:,:,sid] = np.logical_or(canvas[:,:,sid], seg != 0)
        canvas[:,:,sid] = cv2.dilate(canvas[:,:,sid], dilate_kernel)
    canvas[:,:,2] = np.logical_and(canvas[:,:,0], canvas[:,:,1])

    if np.sum(canvas[:,:,2]) > 0:
        print(f"**** Zeroing overlapping pixels, count: {np.sum(canvas[:,:,2])}")

    # zero the overlap region
    acceptPlanes[canvas[:,:,2] != 0, :] = 0
    excludePlane = np.logical_or(excludePlane, canvas[:,:,2])

cv2.namedWindow("viewer", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("viewer", mouseEvent)

quit = False
dirtyImage = True
initFrame = True

frameDirectory = {}

maxFrame = -1
numLabeled = -1
numProofed = -1
def recalculateNumLabeled():
    global numLabeled
    global numProofed
    numLabeled = 0
    numProofed = 0
    for fr in frameDirectory.keys():
        if frameDirectory[fr]["already_labeled"]:
            numLabeled += 1
        if frameDirectory[fr]["already_proofed"]:
            numProofed += 1

def generateFrameDirectory():
    global maxFrame
    input_frames = list(sorted(glob.glob(os.path.join(input_path, "**/*.jpg"), recursive = True)))
    input_frames = list(filter(lambda x: "segmented" not in x, input_frames))
    input_frames = list(filter(lambda x: "proofed" not in x, input_frames))

    # sort by frame number (ascending)
    framenums = sorted(list(map(lambda x: int(os.path.splitext(os.path.split(x)[1])[0]), input_frames)))
    framenums = np.unique(np.array(framenums))
    framenum_to_fdindex = {}
    for f in framenums:
        framenum_to_fdindex[f] = np.where(framenums == f)[0][0]

    for inframe in input_frames:
        inframe_base = os.path.split(os.path.splitext(inframe)[0])[1]
        framenum = int(inframe_base)
        itemid = framenum_to_fdindex[framenum]
        frameDirectory[itemid] = {}
        frameDirectory[itemid]["image_path"] = inframe
        frameDirectory[itemid]["imagefilename_base"] = inframe_base
        frameDirectory[itemid]["segmented_path"] = os.path.join(input_path,
                                                                "segmented", f"{inframe_base}.png")
        frameDirectory[itemid]["proofed_path"] = os.path.join(input_path,
                                                              "proofed", f"{inframe_base}.png")
        frameDirectory[itemid]["framenum"] = int(inframe_base)
        frameDirectory[itemid]["already_labeled"] = False
        frameDirectory[itemid]["already_proofed"] = False
        if os.path.exists(frameDirectory[itemid]['segmented_path']):
            frameDirectory[itemid]["already_labeled"] = True
        if os.path.exists(frameDirectory[itemid]['proofed_path']):
            frameDirectory[itemid]["already_proofed"] = True
        itemid += 1
        if int(inframe_base) > maxFrame:
            maxFrame = int(inframe_base)
    recalculateNumLabeled()

generateFrameDirectory()

labelIndex = -1
def getNextFrameToLabel():
    global labelIndex
    labelIndex += 1
    if labelIndex >= len(frameDirectory.keys()):
        print("Looped frameDirectory")
        labelIndex = 0
    if labelIndex == -1:
        labelIndex = len(frameDirectory.keys()) - 1
    frameinfo = frameDirectory[labelIndex]
    if os.path.exists(frameinfo['segmented_path']):
        frameinfo["already_labeled"] = True
    if os.path.exists(frameinfo['proofed_path']):
        frameinfo["already_proofed"] = True
    return frameinfo

def findNextLabeledFrame(idx):
    idx += 1
    while idx < len(frameDirectory.keys()):
        if frameDirectory[idx]["already_labeled"] == True:
            return idx
        idx += 1
    # none left, loop.
    print("* Wrapped, while searching for another labeled frame to review.")
    return 0

def restoreSavedSegments(frameinfo):
    if os.path.exists(frameinfo["segmented_path"]):
        segmented = cv2.imread(frameinfo["segmented_path"])
        # *B* G *R*
        blue_segment = (segmented[:,:,0] == 255)
        red_segment = (segmented[:,:,2] == 255)
        return [blue_segment, red_segment]

def loadProofedSegments(frameinfo):
    if os.path.exists(frameinfo["proofed_path"]):
        segmented = cv2.imread(frameinfo["proofed_path"])
        # *B* G *R*
        blue_segment = (segmented[:,:,0] == 255)
        red_segment = (segmented[:,:,2] == 255)
        return [blue_segment, red_segment]

screenshot = False

while not quit:
    frameinfo = getNextFrameToLabel()

    print(f"Label object {frameinfo}")

    frame = cv2.imread(frameinfo["image_path"])
    colorframe = cv2.resize(frame, None, fx = 0.5, fy = 0.5)
    print("Pausing")

    saved_masks = [np.zeros((colorframe.shape[0], colorframe.shape[1])),
                   np.zeros((colorframe.shape[0], colorframe.shape[1]))]
    acceptPlanes = np.zeros((colorframe.shape[0], colorframe.shape[1], 2), dtype=np.uint8)
    excludePlane = np.zeros((colorframe.shape[0], colorframe.shape[1]), dtype=np.uint8)

    if frameinfo["already_proofed"]:
        saved_masks = loadProofedSegments(frameinfo)
        acceptRightSizeContours()
        excludeMasksThatOverlap()
    elif frameinfo["already_labeled"]:
        saved_masks = restoreSavedSegments(frameinfo)
        acceptRightSizeContours()
        excludeMasksThatOverlap()

    if initFrame:
        cv2.resizeWindow("viewer", colorframe.shape[1], colorframe.shape[0])
        initFrame = False

    while True:
        if dirtyImage:
            if screenshot:
                do_screenshot(colorframe, frameinfo)
                screenshot = False
            showAll(colorframe, frameinfo)
            dirtyImage = False

        k = cv2.waitKey(25)
        if k == 27:
            #escape
            print("quitting")
            quit = True
            break
        if k == 32:
            #space - next image
            print("next image")
            pickedSegments = []
            pickedSegmentIds = []
            dirtyImage = True
            break
        if k == ord('n'):
            # jump to next labeled frame
            labelIndex = findNextLabeledFrame(labelIndex) - 1
            pickedSegments = []
            pickedSegmentIds = []
            dirtyImage = True
            break
        if k == ord('p'):
            #prev image
            print("next image")
            dirtyImage = True
            labelIndex -= 2
            break
        if k == ord('r'):
            print("resetting image")
            saved_masks = [np.zeros((colorframe.shape[0], colorframe.shape[1])),
                        np.zeros((colorframe.shape[0], colorframe.shape[1]))]
            acceptPlanes = np.zeros((colorframe.shape[0], colorframe.shape[1], 2), dtype=np.uint8)
            excludePlane = np.zeros((colorframe.shape[0], colorframe.shape[1]), dtype=np.uint8)
            saved_masks = restoreSavedSegments(frameinfo)
            acceptRightSizeContours()
            excludeMasksThatOverlap()
            dirtyImage = True
        if k == ord('o'):
            objectId += 1
            objectId %= 3
            dirtyImage = True
            print(f"Brush now assigned to {objectId}")
        if k == ord('v'):
            cycleDrawOptions()
            print(f"Current draw settings: {draw_option}")
            dirtyImage = True
        if k == ord('/'):
            screenshot = True
            dirtyImage = True
            import time
            time.sleep(1)

        if k == ord("d"):
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            if os.path.exists(frameinfo["proofed_path"]):
                os.unlink(frameinfo["proofed_path"])
            # write masks
            print("Deleted proof.")
            frameinfo["already_proofed"] = False
            recalculateNumLabeled()
            dirtyImage = True

        if k == ord("w"):
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            if os.path.exists(frameinfo["proofed_path"]):
                os.unlink(frameinfo["proofed_path"])
            # write masks
            if np.sum(acceptPlanes) == 0:
                print("Not writing empty masks structure.")
                frameinfo["already_proofed"] = False
            else:
                outimage = np.zeros((acceptPlanes.shape[0], acceptPlanes.shape[1], 3), dtype=np.uint8)
                outimage[:,:,0] = (acceptPlanes[:,:,0] != 0) * 255
                outimage[:,:,2] = (acceptPlanes[:,:,1] != 0) * 255
                cv2.imwrite(frameinfo["proofed_path"], outimage)
                frameinfo["already_proofed"] = True
            recalculateNumLabeled()
            print("Saved!")
            dirtyImage = True

print("Quitting")
