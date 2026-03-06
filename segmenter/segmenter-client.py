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

root = tkinter.Tk()
root.withdraw()

#simplified
font = cv2.FONT_HERSHEY_TRIPLEX
font_size = 1
font_thickness = 2
font_backing = False
font_main_color = (140, 30, 70)

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", default="")
parser.add_argument("--standalone", action="store_true")
parser.add_argument("--standalone_model", default="sam_vit_h_4b8939.pth")
parser.add_argument("--standalone_device_option", default="cuda")

script_dir = os.path.dirname(os.path.abspath(__file__))

args = parser.parse_args()
input_path = args.input_path

if input_path == "":
    input_path = os.path.abspath(tkinter.filedialog.askdirectory(title = "Please select the input image folder",
                                                                 initialdir = os.getcwd()))

# check to see if directory contains child pngs somewhere
print(os.path.join(input_path, "**/*"))
childimages = glob.glob(os.path.join(input_path, "**/*"), recursive = True)
childimages = list(filter(lambda x: os.path.splitext(os.path.split(x)[1])[1] in [".png", ".jpg"],
                    childimages))
childimages = list(filter(lambda x: "segmented" not in x, childimages))
childimages = list(filter(lambda x: "proofed" not in x, childimages))
childimages = list(filter(lambda x: "verification" not in x, childimages))
if len(childimages) == 0:
    print("Expected path to contain input images! Quitting.")
    sys.exit(1)

input_path = os.path.join(os.path.split(childimages[0])[0])
output_path = os.path.join(os.path.split(childimages[0])[0], "segmented")
print("Input path:")
print(input_path)
print("Output path:")
print(output_path)

if not os.path.exists(output_path):
    os.makedirs(output_path)

samPoints = []
samPointIds = []
pickedSegments = []
pickedSegmentIds = []
candidateSegment = None

anColorsSaved = [[210, 85, 25], [120, 80, 225], [120, 120, 0]]
anColors = [[150, 55, 10], [80, 60, 175], [60, 60, 0]]
uiColors = [[128, 128, 0], [0, 128, 128], [180, 20, 180]]

# client context for remote and standalone SAM inference
predictor = None
runpod_SAM = None

if args.standalone:
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=os.path.join(script_dir, args.standalone_model))
    sam.to(device=args.standalone_device_option)
    predictor = SamPredictor(sam)
else:
    import runpod
    with open(os.path.join(script_dir, "runpod_client_config.json"), "r") as fh:
        cfg = json.load(fh)
        print(f"Using key id: {cfg['key_id']}")
        runpod.api_key = cfg["api_key"]
        runpod_SAM = runpod.Endpoint(cfg["endpoint"])

    print(f"Attempting to connect to runpod instance endpoint={cfg['endpoint']}...")
    job = runpod_SAM.run({ "input": { "verb": "health" }})
    s = job.status()
    while s in ["IN_PROGRESS", "IN_QUEUE"]:
        print(f"Remote health-check status: {s}")
        time.sleep(5)
        s = job.status()
    print(f"Test job final status: {s}")
    print(job.output())

def predict_SAM_remote(colorframe, point_coords, point_labels):
    payload = {}
    payload["input"] = {}
    payload["input"]["verb"] = "predict"
    payload["input"]["samPoints"] = list(point_coords)
    payload["input"]["point_labels"] = list(point_labels)
    payload["input"]["colorframe"] = {
        "data": "<omited for legibility>",
        "height": colorframe.shape[0],
        "width": colorframe.shape[1],
    }
    print(json.dumps(payload))
    payload["input"]["colorframe"]["data"] = base64.b64encode(colorframe).decode("ascii")

    output = runpod_SAM.run_sync(payload)
    samOutputMasks = np.frombuffer(base64.b64decode(output["samOutputMasks"]["data"]), dtype=np.uint8)
    samOutputMasks = samOutputMasks.reshape(output["samOutputMasks"]["count"],
                                            output["samOutputMasks"]["height"],
                                            output["samOutputMasks"]["width"])
    return samOutputMasks

def predict_SAM_standalone(colorframe, point_coords, point_labels):
    predictor.set_image(colorframe)
    samOutputMasks, _, _ = predictor.predict(point_coords = np.array(point_coords),
                                                 point_labels = point_labels)
    return samOutputMasks

textColor = (150, 80, 60)

def overlaySegments(colorframe):
    overlayed = colorframe.copy()
    for overlayIdx in range(len(previouslySavedSegmentIds)):
        previouslySavedSegment = previouslySavedSegments[overlayIdx]
        previouslySavedSegment_inv = cv2.bitwise_not(previouslySavedSegment)
        previouslySavedSegmentId = previouslySavedSegmentIds[overlayIdx]
        palette = np.zeros((overlayed.shape[0], overlayed.shape[1], 3), dtype=np.uint8)
        palette[:,:] = np.array(anColorsSaved[previouslySavedSegmentId])
        palette = cv2.bitwise_and(palette, palette, mask=previouslySavedSegment)
        overlayed = cv2.add(overlayed, palette)
    for overlayIdx in range(len(pickedSegmentIds)):
        pickedSegment = pickedSegments[overlayIdx]
        pickedSegmentId = pickedSegmentIds[overlayIdx]
        print(f"Overlaying object with dimensions: {pickedSegment.shape}")
        overlayed[pickedSegment != 0, 0] += np.uint8(anColorsSaved[pickedSegmentId][0] * 0.3)
        overlayed[pickedSegment != 0, 1] += np.uint8(anColorsSaved[pickedSegmentId][1] * 0.3)
        overlayed[pickedSegment != 0, 2] += np.uint8(anColorsSaved[pickedSegmentId][2] * 0.3)
    if not candidateSegment is None:
        print("Drawing an overlay candidate segment")

        # add a bunch to the 'green' channel
        overlayed[candidateSegment != 0, 1] += 80
    return overlayed

def consolidateMasks():
    outputs = {}
    #for sidx in range(len(pickedSegments)):
    #    print(f"Consolidate Mask {sidx}: id: {pickedSegmentIds[sidx]} size: {np.sum(pickedSegments[sidx])}")
    for sidx in range(len(pickedSegments)):
        segid = pickedSegmentIds[sidx]
        seg = pickedSegments[sidx]
        if segid not in outputs:
            outputs[segid] = (seg * 255).astype(np.uint8)
        else:
            outputs[segid] = (np.logical_or(seg != 0, outputs[segid] != 0) * 255).astype(np.uint8)
                             #(((seg * 1).astype(np.uint8) + outputs[segid]) != 0) * 255
    for oid in outputs.keys():
        print(f"Output Masks {oid} - size {np.sum(outputs[oid] != 0)}")
    return list(outputs.values())

def showAll(colorframe, frameinfo):
    cv2.imshow("LabelMasks", frameWithMarkup(overlaySegments(colorframe), frameinfo))

def saveAll(colorframe, frameinfo):
    import time
    fname = f"screenshot-{int(time.time())}.png"
    cv2.imwrite(fname, frameWithMarkup(overlaySegments(colorframe), frameinfo))

objectId = 0

def losePreviouslySaved():
    global previouslySavedSegments
    global previouslySavedSegmentIds
    previouslySavedSegments = []
    previouslySavedSegmentIds = []

def mouseEvent(event, x, y, flags, param):
    global dirtyImage
    global samPoints
    global samPoints
    global samPointIds
    if (event == cv2.EVENT_LBUTTONDOWN):
        samPoints.append([x, y])
        samPointIds.append(objectId)
        losePreviouslySaved()
        dirtyImage = True

def frameWithMarkup(colorframe, frameinfo):
    for ptidx in range(len(samPoints)):
        pt = samPoints[ptidx]
        ptid = samPointIds[ptidx]
        colorframe = cv2.circle(colorframe, pt, 12,
                                            [255, 255, 255], -1)
        colorframe = cv2.circle(colorframe, pt, 9,
                                            [int(anColorsSaved[ptid][0]*0.9),
                                             int(anColorsSaved[ptid][1]*0.9),
                                             int(anColorsSaved[ptid][2]*0.9)],
                                            -1)
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
    for phase in ["labeled", "reticule"]:
        for i in frameDirectory.keys():
            f = frameDirectory[i]["framenum"]
            labeled = frameDirectory[i]["already_labeled"]

            if phase == "labeled":
                height = 7
                if labeled:
                    color = uiColors[0]
                else:
                    color = uiColors[1]
                colorframe = cv2.line(colorframe,
                                (int(f/maxFrame * labeledBarXWidth + labeledBarXBase), int(labeledBarYBase-height)),
                                (int(f/maxFrame * labeledBarXWidth + labeledBarXBase), int(labeledBarYBase+height)),
                                color, 3)

            if phase == "reticule" and f == frameinfo["framenum"]:
                # Display a reticule to indicate current frame
                height = 15
                if labeled:
                    color = uiColors[0]
                else:
                    color = uiColors[1]
                colorframe = cv2.line(colorframe,
                                (int(f/maxFrame * labeledBarXWidth + labeledBarXBase), int(labeledBarYBase-height)),
                                (int(f/maxFrame * labeledBarXWidth + labeledBarXBase), int(labeledBarYBase+height)),
                                color, 5)

    curframe = frameinfo["framenum"]
    percent = int(curframe / maxFrame * 100)
    colorframe = cv2.putText(colorframe,
                             f"Frame: {curframe} / {maxFrame} [{percent:02d}%].  {numLabeled} labeled frames.",
                             (70, 40),
                             font, font_size, font_main_color, font_thickness)
    print(f"Frame: {curframe} / {maxFrame} [{percent:02d}%].  {numLabeled} labeled frames.")

    return colorframe


cv2.namedWindow("LabelMasks", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("LabelMasks", mouseEvent)

quit = False
dirtyImage = True
initFrame = True

frameDirectory = {}

maxFrame = -1
numLabeled = -1
def recalculateNumLabeled():
    global numLabeled
    numLabeled = 0
    for fr in frameDirectory.keys():
        if frameDirectory[fr]["already_labeled"]:
            numLabeled += 1

def generateFrameDirectory():
    global maxFrame
    input_frames = list(sorted(glob.glob(os.path.join(input_path, "*.jpg"))))
    input_frames = list(filter(lambda x: "verification" not in x, input_frames))

    itemid = 0
    for inframe in input_frames:
        inframe_base = os.path.split(os.path.splitext(inframe)[0])[1]
        frameDirectory[itemid] = {}
        frameDirectory[itemid]["image_path"] = inframe
        frameDirectory[itemid]["imagefilename_base"] = inframe_base
        frameDirectory[itemid]["segmented_path"] = os.path.join(output_path, f"{inframe_base}.png")
        frameDirectory[itemid]["framenum"] = int(inframe_base)
        frameDirectory[itemid]["already_labeled"] = False
        if os.path.exists(frameDirectory[itemid]['segmented_path']):
            frameDirectory[itemid]["already_labeled"] = True
        itemid += 1
        if int(inframe_base) > maxFrame:
            maxFrame = int(inframe_base)
    recalculateNumLabeled()

generateFrameDirectory()

labelIndex = -1
def getNextFrameToLabel(randomSeek):
    global labelIndex
    if randomSeek:
        labelIndex = int(random.random() * len(frameDirectory.keys()))
    else:
        labelIndex += 1
        if labelIndex >= len(frameDirectory.keys()):
            print("Looped frameDirectory")
            labelIndex = 0
        if labelIndex == -1:
            labelIndex = len(frameDirectory.keys()) - 1
    frameinfo = frameDirectory[labelIndex]
    if os.path.exists(frameinfo['segmented_path']):
        frameinfo["already_labeled"] = True
    return frameinfo

def restoreSavedSegments(frameinfo):
    global previouslySavedSegments
    global previouslySavedSegmentIds
    for sid in range(2):
        if os.path.exists(frameinfo["segmented_path"]):
            segmented = cv2.imread(frameinfo["segmented_path"])
            # *B* G *R*
            blue_segment = (segmented[:,:,0] == 255).astype(np.uint8)
            previouslySavedSegments.append(blue_segment)
            previouslySavedSegmentIds.append(0)
            red_segment = (segmented[:,:,2] == 255).astype(np.uint8)
            previouslySavedSegments.append(red_segment)
            previouslySavedSegmentIds.append(1)

saveFrame = False

randomSeek = False
while not quit:
    frameinfo = getNextFrameToLabel(randomSeek)
    randomSeek = False

    print(f"Label object {frameinfo}")

    frame = cv2.imread(frameinfo["image_path"])
    print("Pausing")

    candidateSegment = None
    samOutputMasks = []
    selectedSegments = []
    selectedSegmentIds = []
    pickedSegments = []
    pickedSegmentIds = []
    previouslySavedSegments = []
    previouslySavedSegmentIds = []

    if frameinfo["already_labeled"]:
        restoreSavedSegments(frameinfo)

    frame = cv2.blur(frame, (5,5))

    colorframe = cv2.resize(frame, None, fx = 0.5, fy = 0.5)
    if initFrame:
        cv2.resizeWindow("LabelMasks", colorframe.shape[1], colorframe.shape[0])
        initFrame = False

    while True:
        if dirtyImage:
            if saveFrame:
                saveAll(colorframe, frameinfo)
                saveFrame = False
            showAll(colorframe, frameinfo)
            dirtyImage = False

        k = cv2.waitKey(25)
        if k == 27:
            #escape
            quit = True
            break
        if k == 32:
            #space - next image
            print("next image")
            samPoints = []
            samPointIds = []
            pickedSegments = []
            pickedSegmentIds = []
            dirtyImage = True
            break
        if k == ord('p'):
            #prev image
            print("next image")
            samPoints = []
            samPointIds = []
            pickedSegments = []
            pickedSegmentIds = []
            dirtyImage = True
            labelIndex -= 2
            break
        if k == ord('r'):
            print("resetting image")
            samPoints = []
            samPointIds = []
            pickedSegments = []
            pickedSegmentIds = []
            selectedSegments = []
            selectedSegmentIds = []
            previouslySavedSegments = []
            previouslySavedSegmentIds = []
            samOutputMasks = []
            curSamOutput = -1
            dirtyImage = True
        if k == ord('o'):
            objectId += 1
            objectId %= 3
            dirtyImage = True
            print(f"Now labeling / predicting {objectId}")
        if k == ord('='):
            randomSeek = True
            samPoints = []
            samPointIds = []
            pickedSegments = []
            pickedSegmentIds = []
            dirtyImage = True
            break
        if k == ord('/'):
            saveFrame = True
            dirtyImage = True
            import time
            time.sleep(1)
        if k == ord('s'):
            if len(samPoints) == 0:
                print("Refusing to run SAM with 0 points selected.")
                continue
            print("Running SAM")
            # run SAM on the SAM mask?

            candidateSegment = None
            print("running predictor:")
            point_labels = list(map(lambda x: 1 if x == objectId else 0, samPointIds))
            print(f"Point assignments: {point_labels}")
            print(f"Segmenting with {len(samPoints)} points")

            if args.standalone:
                samOutputMasks = predict_SAM_standalone(colorframe,
                                                        point_coords = samPoints,
                                                        point_labels = point_labels)
            else:
                samOutputMasks = predict_SAM_remote(colorframe,
                                                    point_coords = samPoints,
                                                    point_labels = point_labels)

            print(f"Masks found: {len(samOutputMasks)}")
            curSamOutput = -1
            k = ord('d')

        if k == ord("d"):
            if len(samOutputMasks) > 0:
                print("Cycling through sam masks")
                # cycle through SAM outputs
                curSamOutput += 1
                curSamOutput %= len(samOutputMasks)
                candidateSegment = samOutputMasks[curSamOutput]
                print(f"Candidate segment dimensions: {candidateSegment.shape}")
                dirtyImage = True

        if k == ord("1"):
            # name result as "0"
            pickedSegments.append(samOutputMasks[curSamOutput])
            pickedSegmentIds.append(0)
            candidateSegment = None
            print("Picked as '1'")
            samOutputMasks = []
            dirtyImage = True
        if k == ord("2"):
            pickedSegments.append(samOutputMasks[curSamOutput])
            pickedSegmentIds.append(1)
            candidateSegment = None
            print("Picked as '2'")
            samOutputMasks = []
            dirtyImage = True

        if k == ord("w"):
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            consolidatedMasks = consolidateMasks()
            if os.path.exists(frameinfo["segmented_path"]):
                os.unlink(frameinfo["segmented_path"])
            # write masks
            if np.sum(consolidatedMasks) == 0:
                print("Not writing empty masks structure.")
                frameinfo["already_labeled"] = False
            else:
                outimage = np.zeros((list(consolidatedMasks[0].shape) + [3]))
                outimage[:,:,0] = (consolidatedMasks[0] != 0) * 255
                outimage[:,:,2] = (consolidatedMasks[1] != 0) * 255
                cv2.imwrite(frameinfo["segmented_path"], outimage)
                frameinfo["already_labeled"] = True
            recalculateNumLabeled()
            print("Saved!")

print("Quitting")
