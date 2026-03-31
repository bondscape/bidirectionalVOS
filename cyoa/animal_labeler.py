import os
import cv2
import sys
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", required=True)
parser.add_argument("--label_path", default="", required=False)
args = parser.parse_args()

video_path = args.video_path
label_path = args.label_path
if label_path == "":
    label_path = video_path + ".identities.json"

clickEventFired = False
mousePointX = -1
mousePointY = -1

def mouseEvent(event, x, y, flags, param):
    global seekEventFired
    global clickEventFired
    global mousePointX
    global mousePointY
    if (event == cv2.EVENT_LBUTTONDOWN):
        mousePointX = x
        mousePointY = y
        clickEventFired = True

cv2.namedWindow("main", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("main", mouseEvent)

def getAnimalIdentities(label_path):
    if not os.path.exists(label_path):
        return { "identities": {}}
    with open(label_path, "r") as fh:
        identities = json.loads(fh.read())
        return identities

def saveAnimalIdentities(identities, label_path):
    with open(label_path, "w+") as fh:
        dump = {}
        dump["identities"] = identities
        fh.write(json.dumps(dump, indent=4))

def LabelFrame(video_path, label_path):
    global clickEventFired
    global mousePointX

    vidfile = cv2.VideoCapture(video_path)
    if not vidfile.isOpened():
        print(f"***** Unable to open {video}")

    _, frame = vidfile.read()
    orig_height = frame.shape[0]
    orig_width = frame.shape[1]
    tgt_height = int(orig_height / 2)
    tgt_width = int(orig_width / 2)
    cv2.resizeWindow("main", tgt_width, tgt_height)
    disableOverlay = False

    sex = "female"
    labeledAnimals = {"female": [], "male": []}

    labeledAnimals = getAnimalIdentities(label_path)["identities"]
    print(f"Animal identities: {labeledAnimals}")

    dataChanged = False

    def showFrame(frame):
        frame = frame.copy()
        if disableOverlay:
            cv2.imshow("main", frame)
            return

        descriptorText = f"Video name: {os.path.basename(video_path)}"
        sexColors = {
                       "female": [202, 151, 237],
                       "male": [206, 158, 114],
                   }

        annotationColor = (120, 30, 100)

        frame = cv2.putText(frame, descriptorText, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, annotationColor, 2)
        savedStatus = "Work not saved - '/' to save and move to next video."
        savedStatusColor = [40, 60, 200]
        if dataChanged == False:
            savedStatus = ""

        frame = cv2.putText(frame, savedStatus, (50, frame.shape[0]-150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, savedStatusColor, 2)

        frame = cv2.circle(frame, (frame.shape[1]-50, 50), 20, sexColors[sex], -1)

        for s in ["female", "male"]:
            if s in labeledAnimals:
                if len(labeledAnimals[s]) == 0:
                    continue
                adjustedCoordinates = [int(labeledAnimals[s][0] * tgt_width / orig_width),
                                       int(labeledAnimals[s][1] * tgt_height / orig_height)]
                frame = cv2.circle(frame, adjustedCoordinates, 5, sexColors[s], -1)

        cv2.imshow("main", frame)

    ret, firstFrame = vidfile.read()
    if not ret:
        print(f"Unable to read frame from {video_path}!")
        sys.exit(1)

    dirtyFrame = True
    while True:
        frame = cv2.resize(frame, (tgt_width, tgt_height))

        while True:
            if clickEventFired:
                labeledAnimals[sex] = [int(orig_width * mousePointX / tgt_width),
                                     int(orig_height * mousePointY / tgt_height)]
                clickEventFired = False
                dirtyFrame = True
                dataChanged = True

            if dirtyFrame:
                showFrame(frame)
                dirtyFrame = False

            key = cv2.waitKey(100)
            if key == 27:
                print("Quit!")
                sys.exit(1)
            if key == ord('\t'):
                disableOverlay = not disableOverlay
                dirtyFrame = True
            if key == ord('f'):
                sex = "female"
                dirtyFrame = True
            if key == ord('m'):
                sex = "male"
                dirtyFrame = True
            if key == ord('/'):
                print("Save processing instructions!")
                for s in ["female", "male"]:
                    identities = {}
                    identities["female"] = labeledAnimals["female"]
                    identities["male"] = labeledAnimals["male"]
                saveAnimalIdentities(identities, label_path)
                dataChanged = False
                return False

LabelFrame(video_path, label_path)
