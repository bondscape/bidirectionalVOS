import os
import subprocess
from pathlib import Path
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

def get_video_frame_count(video_path):
    cache_path = Path(f"{video_path}.frame_count")

    if cache_path.exists():
        return int(cache_path.read_text().strip())

    # otherwise...
    print("=== Note: Measuring video length *reliably*, which may take a bit.")
    print("  (Some video containers are not accurate about content lengths)")

    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    count = int(result.stdout.strip())
    cache_path.write_text(f"{count}")
    return count

cv2.namedWindow("main", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("main", mouseEvent)

def getAnimalIdentities(label_path):
    if not os.path.exists(label_path):
        return {"annotation_frame": 0, "identities": {"female": [-1, -1], "male": [-1, -1]}}
    with open(label_path, "r") as fh:
        labels = json.loads(fh.read())
        return labels

def saveAnimalIdentities(labels, label_path):
    with open(label_path, "w+") as fh:
        dump = {}
        dump["identities"] = labels["identities"]
        dump["annotation_frame"] = labels["annotation_frame"]
        fh.write(json.dumps(dump, indent=4))

def LabelFrame(video_path, label_path):
    global clickEventFired
    global mousePointX

    vidfile = cv2.VideoCapture(video_path)
    if not vidfile.isOpened():
        print(f"***** Unable to open {video}")
    fps = vidfile.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = get_video_frame_count(video_path)

    seek_step = int(round(5 * fps))

    sex = "female"
    labeledAnimals = getAnimalIdentities(label_path)
    print(f"Animal identities: {labeledAnimals}")

    dataChanged = False

    def showFrame(frame):
        frame = frame.copy()
        if disableOverlay:
            cv2.imshow("main", frame)
            return

        descText = f"Video name: {os.path.basename(video_path)}, frame: {current_frame}"
        infoText = f"Annotations for frame: {labeledAnimals['annotation_frame']}"
        helpText = f"Controls: space (pause), z/v (back/ffwd), m/f (sex), / (save)"

        sexColors = {
                       "female": [202, 151, 237],
                       "male": [206, 158, 114],
                   }

        annotationColor = (120, 30, 100)

        frame = cv2.putText(frame, descText, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, annotationColor, 2)
        frame = cv2.putText(frame, infoText, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, annotationColor, 2)

        savedStatus = "Work not saved - '/' to save and move to next video."
        savedStatusColor = [40, 60, 200]
        if dataChanged == False:
            savedStatus = ""

        frame = cv2.putText(frame, savedStatus, (50, frame.shape[0]-150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, savedStatusColor, 2)

        frame = cv2.circle(frame, (frame.shape[1]-50, 50), 20, sexColors[sex], -1)

        for s in ["female", "male"]:
            if s in labeledAnimals["identities"]:
                if len(labeledAnimals["identities"][s]) == 0:
                    continue
                adjustedCoordinates = [int(labeledAnimals["identities"][s][0] * tgt_width / orig_width),
                                       int(labeledAnimals["identities"][s][1] * tgt_height / orig_height)]
                frame = cv2.circle(frame, adjustedCoordinates, 5, sexColors[s], -1)

        cv2.imshow("main", frame)

    current_frame = 0
    ret, frame = vidfile.read()
    if not ret:
        print(f"Unable to read frame 0 from {video_path}!")
        sys.exit(1)

    orig_height = frame.shape[0]
    orig_width = frame.shape[1]
    tgt_height = int(orig_height / 2)
    tgt_width = int(orig_width / 2)
    frame_disp = cv2.resize(frame, (tgt_width, tgt_height))
    cv2.resizeWindow("main", tgt_width, tgt_height)

    disableOverlay = False
    paused = True
    dirtyFrame = True

    while True:
        if not paused:
            ret, frame = vidfile.read()
            if not ret:
                vidfile.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_frame = 0
                ret, frame = vidfile.read()
                frame_disp = cv2.resize(frame, (tgt_width, tgt_height))
                paused = True
            else:
                current_frame += 1
                frame_disp = cv2.resize(frame, (tgt_width, tgt_height))
                dirtyFrame = True

        if clickEventFired:
            labeledAnimals["identities"][sex] = \
                                [int(orig_width * mousePointX / tgt_width),
                                 int(orig_height * mousePointY / tgt_height)]
            labeledAnimals["annotation_frame"] = current_frame
            clickEventFired = False
            dirtyFrame = True
            dataChanged = True

        if dirtyFrame:
            showFrame(frame_disp)
            dirtyFrame = False

        delay = 1 if paused else max(1, int(1000 / fps))
        key = cv2.waitKey(1)

        if key == 27:
            print("Quit!")
            sys.exit(1)
        if key == ord('\t'):
            disableOverlay = not disableOverlay
            dirtyFrame = True
        if key == ord(' '):
            paused = not paused
        if key == ord('z'):
            target_frame = max(0, current_frame - seek_step)
            vidfile.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = vidfile.read()
            if not ret:
                print(f"Unable to read frame: {target_frame}")
            else:
                frame_disp = cv2.resize(frame, (tgt_width, tgt_height))
                current_frame = target_frame
            dirty_frame = True
        if key == ord('v'):
            target_frame = min(total_frames-1, current_frame + seek_step)
            vidfile.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = vidfile.read()
            if not ret:
                print(f"Unable to read frame: {target_frame}")
            else:
                frame_disp = cv2.resize(frame, (tgt_width, tgt_height))
                current_frame = target_frame
            dirty_frame = True
        if key == ord('f'):
            sex = "female"
            dirtyFrame = True
        if key == ord('m'):
            sex = "male"
            dirtyFrame = True
        if key == ord('/'):
            print(f"Saving processing instructions with labels: {labeledAnimals}!")
            saveAnimalIdentities(labeledAnimals, label_path)
            dataChanged = False
            return False

LabelFrame(video_path, label_path)
