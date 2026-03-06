import os
import sys
import glob
import tempfile

from omegaconf import OmegaConf, open_dict

import torch
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model as get_default_cutie_model

import cv2
import numpy as np
import time
from PIL import Image
import argparse
import h5py

def combineFrames(direction, output_path, output_masks_h5):
    print(f"Combining frames in for {output_masks_h5} - direction: {direction}")

    # infiles:
    print(f"searching for files: {os.path.join(output_path, '*.png')}")
    infiles = list(sorted(glob.glob(os.path.join(output_path, "*.png"))))

    print(f"First infile: {infiles[0]}")
    print(f"Last infile: {infiles[-1]}")

    if direction == "reverse":
        infiles = infiles[::-1]
        print(f"Post-reversal first infile: {infiles[0]}")
        print(f"Post-reversal last infile: {infiles[-1]}")

    mask = np.array(Image.open(infiles[0]))
    dataset_shape = list([len(infiles)] + list(mask.shape[:2]))
    chunk_size = tuple([1] + list(mask.shape[:2]))

    start_time = time.time()
    with h5py.File(output_masks_h5, 'w') as f:
        dataset = f.create_dataset('masks', shape=dataset_shape,
                                   chunks=chunk_size, dtype='uint8', compression='gzip')
        fnum = 0
        for infile in infiles:
            mask = np.array(Image.open(infile))
            if type(mask) == type(None):
                print(f"******* missing file {infile}")
                break
            dataset[fnum, :, :] = mask.astype(np.uint8)

            if fnum % 500 == 0:
                print(f"Stored mask {fnum} (which was {infile}")
            fnum += 1

    end_time = time.time()
    print(f"Stored {len(infiles)} masks in {end_time-start_time}s @ {(end_time-start_time)/len(infiles)}s/frame")
    return

def process_video(cutie_model, frames_dir, masks_dir, video_path, clip_mask_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    cutie_model = cutie_model.to(device).eval()
    processor = InferenceCore(cutie_model, cfg=cutie_model.cfg)

    print(f"Device arg: {device}")
    print(f"Model is on: {next(cutie_model.parameters()).device}")

    # how many objects are we segmenting?
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    first_mask = np.array(Image.open(mask_files[0]))
    objects = np.unique(first_mask)
    objects = objects[objects != 0].tolist()
    n_objects = len(objects)
    print(f"Found {n_objects} objects to segment in {os.path.split(mask_files[0])[1]}")

    with torch.inference_mode():
        for mask_path in mask_files:
            mask_name = os.path.splitext(os.path.basename(mask_path))[0]
            frame_path = os.path.join(frames_dir, mask_name + ".png")

            print(f"Loading {mask_name} into permanent memory")
            frame = np.array(Image.open(frame_path)).astype(np.float32) / 255.0
            frame_torch = torch.from_numpy(frame).permute(2, 0, 1).to(device)
            print(f"Frame tensor is on: {frame_torch.device}")

            mask = np.array(Image.open(mask_path))
            mask_torch = torch.from_numpy(mask).to(device)
            processor.step(frame_torch, mask_torch, objects=objects, force_permanent=True)

    # run inference on video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}!")
        sys.exit()

    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Begin at the beginning!
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    # convenience palette for testing
    palette = [0] * 768
    palette[0:3] = [0, 0, 0]
    palette[3:6] = [255, 0, 0]
    palette[6:9] = [0, 255, 0]

    first_frame = True

    with torch.inference_mode():
        while cap.isOpened():
            if frame_idx % 100 == 0:
                print(f"Frame {frame_idx} / {total_frame_count}")

            _, frame = cap.read()
            if frame is None or frame_idx > total_frame_count:
                print("Completed reading input file (or failed to read another frame)")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frame_torch = torch.from_numpy(frame).permute(2, 0, 1).to(device)

            if first_frame:
                first_frame = False
                print(f"Loading predefined mask for frame: {frame_idx:07d}")
                predefined_mask_path = os.path.join(clip_mask_dir, f"{frame_idx:07d}.png")
                print(f"Loading {predefined_mask_path} as {frame_idx:07d}")
                predefined_mask = np.array(Image.open(predefined_mask_path))
                mask_torch = torch.from_numpy(predefined_mask).to(device)

                # feed our defined mask into the model
                output_prob = processor.step(frame_torch, mask_torch, objects=objects)
            else:
                output_prob = processor.step(frame_torch)

            out_mask = processor.output_prob_to_mask(output_prob)
            out_img = Image.fromarray(out_mask.cpu().numpy().astype(np.uint8))
            out_img = out_img.convert('P')
            out_img.putpalette(palette)
            out_img.save(os.path.join(output_dir, f"{frame_idx:07d}.png"))
            frame_idx += 1

    cap.release()
    print("Completed processing")

def videoExists(path):
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < 500000:
        return False
    vid = cv2.VideoCapture(path)
    if not vid.isOpened():
        return False
    return True

def reverse_video(inpath, outpath):
    cap = cv2.VideoCapture(inpath)
    if not cap.isOpened():
        print(f"Failed to open {inpath}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Reversing {nframes} frames from {inpath}")

    with tempfile.TemporaryDirectory() as tmpdir:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(tmpdir, f"{idx:07d}.jpg"), frame)
            idx += 1
        cap.release()

        # reassemble in reverse
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(outpath, fourcc, fps, (width, height))
        for i in range(idx - 1, -1, -1):
            frame = cv2.imread(os.path.join(tmpdir, f"{i:07d}.jpg"))
            writer.write(frame)
        writer.release()

    print(f"Wrote reversed video to {outpath} nframes:[{idx}] vs total:[{nframes}]")

def saferm(path):
    if os.path.exists(path):
        os.unlink(path)

parser = argparse.ArgumentParser()
parser.add_argument("--video", help="full video file", required=True)
parser.add_argument("--clipvideo", help="file name for selected video clip", required=True)
parser.add_argument("--clip_mask_dir", help="Directory with mask files [0000000.png]", required=True)
parser.add_argument("--common_mask_dir", help="Directory with masks [0000000.png]", required=True)
parser.add_argument("--common_frames_dir", help="Directory with frames [0000000.jpg]", required=True)
parser.add_argument("--output_dir", help="Cutie output masks")
parser.add_argument("--device", help="Inference device: [cuda, cpu]", default="cuda")

parser.add_argument("--direction", default="forward", help="forward or reverse")
parser.add_argument("--start_frame", help="start frame for clip extraction", type=int)
parser.add_argument("--end_frame", help="end frame for clip extraction", type=int)
args = parser.parse_args()

# generate and optionally reverse the video based on direction of inference
begin = int(args.start_frame)
end = int(args.end_frame)

clipvideo = args.clipvideo
direction = args.direction

print(f"scanning for {args.clipvideo} if does not exist")
if direction == "forward":
    if not videoExists(args.clipvideo):
        saferm(args.clipvideo)
        print(f"ffmpeg -i \"{args.video}\" -vf select=\"between(n\\,{begin}\\,{end})\" \"{args.clipvideo}\"")
        os.system(f"ffmpeg -i \"{args.video}\" -vf select=\"between(n\\,{begin}\\,{end})\" \"{args.clipvideo}\"")
    clipvideo_forward = clipvideo
elif direction == "reverse":
    clipvideo_forward = args.clipvideo + ".forward.mp4"
    if not videoExists(args.clipvideo) or not videoExists(clipvideo_forward):
        saferm(args.clipvideo)
        saferm(clipvideo_forward)
        print(f"ffmpeg -i \"{args.video}\" -vf select=\"between(n\\,{begin}\\,{end})\" \"{clipvideo_forward}\"")
        os.system(f"ffmpeg -i \"{args.video}\" -vf select=\"between(n\\,{begin}\\,{end})\" \"{clipvideo_forward}\"")
        print(f"ffmpeg -i {clipvideo_forward} -vf reverse {args.clipvideo}")
        #os.system(f"ffmpeg -i {clipvideo_forward} -vf reverse {args.clipvideo}")
        reverse_video(clipvideo_forward, args.clipvideo)
else:
    print(f"unknown direction: {args.direction}")
    sys.exit(1)

# load and customize cutie
cutie_model = get_default_cutie_model()
overrides = OmegaConf.load("default_video_config.yaml")
with open_dict(cutie_model.cfg):
    cutie_model.cfg = OmegaConf.merge(cutie_model.cfg, overrides)

print(f"Performing cutie inference forward from: {args.clipvideo}")
process_video(cutie_model, args.common_frames_dir, args.common_mask_dir, args.clipvideo, args.clip_mask_dir, args.output_dir, args.device)

clipvideobase = os.path.splitext(os.path.split(clipvideo)[1])[0]

combineFrames(direction, args.output_dir,
              os.path.join(args.output_dir, f"{clipvideobase}.masks.h5"))

# clean up the now combined .png files
for f in glob.glob(os.path.join(args.output_dir, "*.png")):
    os.remove(f)
with open(os.path.join(args.output_dir, "masks.combined"), "a") as fh:
    pass
print(f"Done! {clipvideobase}-{direction}")
