#!/usr/bin/python3

import os
import sys
import cv2
import argparse
import numpy as np

def write_job_complete(outpath, success):
    with open(os.path.join(outpath, "thumbnails.complete"), "w") as f:
        if success:
            f.write("Job success.\n")
        else:
            f.write("Job failure.\n")

parser = argparse.ArgumentParser()
parser.add_argument("--video", required = True)
parser.add_argument("--output_path", required = True)
parser.add_argument("--start_frame", default = "0")
parser.add_argument("--end_frame", default = "-1")
parser.add_argument("--frames_to_output", default = "250")
args = parser.parse_args()

infile = os.path.abspath(args.video)
output_dir = os.path.abspath(args.output_path)
os.makedirs(output_dir, exist_ok=True)

print(f"Opening {infile} for frame generation")
cap = cv2.VideoCapture(infile)

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if n_frames <= 0:
    print(f"Failed to open input video!")
    write_job_complete(output_dir, False)
    sys.exit(1)

frames_to_output = int(args.frames_to_output)

start_frame = int(args.start_frame)
end_frame = int(args.end_frame)
if end_frame < 0:
    end_frame = int(n_frames) - 1

print(f"Nframes: {n_frames}")
print(f"start: {start_frame}")
print(f"end_frame: {end_frame}")
print(f"frames to output: {frames_to_output}")

frames_output = 0

target_frames = np.linspace(start_frame, end_frame, frames_to_output, dtype=int)

frame_number = 0
while frames_output < frames_to_output:
    ret, frame = cap.read()
    if ret == False:
        print(f"Job failed! Incomplete read of file {infile}")
        write_job_complete(output_dir, False)
        sys.exit(1)
    if frame_number in target_frames:
        if frames_output % 5 == 0:
            print(f"On video frame number {frame_number}. Frames output: {frames_output} / {frames_to_output}")
        cv2.imwrite(os.path.join(output_dir, f"{frame_number:07d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
        frames_output += 1
    frame_number += 1

print("Done generating thumbnails!")
cap.release()
write_job_complete(output_dir, True)
