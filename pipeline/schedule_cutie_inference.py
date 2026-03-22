import os
import cv2
from PIL import Image
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--video",
                    help="Path to input_video.mp4",
                    required=True)
parser.add_argument("--thumbnail_path",
                    help="Path to directory containing segmented+proofed thumbnails",
                    required=True)
parser.add_argument("--output_path",
                    help="Working directory for cutie inference",
                    required=True)
args = parser.parse_args()

def local_list_dir(dir, entryType = None):
    output = []
    with os.scandir(dir) as it:
        for entry in it:
            if entry.name.startswith("."):
                continue
            if entryType == "DIR" and entry.is_dir():
                output.append(entry.name)
            if entryType == "FILE" and entry.is_file():
                output.append(entry.name)
    return output

def convertSegmentedToCutieIdentities(inpath, outpath):
    inimage = cv2.imread(inpath)
    outimage = np.zeros(list(inimage.shape)[0:2], dtype=np.uint8)
    outimage[inimage[:,:,0] != 0] = 1
    outimage[inimage[:,:,2] != 0] = 2
    outimage[inimage[:,:,1] != 0] = 3
    outimage = cv2.resize(outimage, (0, 0), fx = 2, fy = 2, interpolation=cv2.INTER_NEAREST_EXACT)
    import matplotlib.pyplot as plt

    img = Image.fromarray(outimage, mode='L')
    img = img.convert('P')
    palette = [0] * 768
    palette[0:3] = [0, 0, 0]
    palette[3:6] = [255, 0, 0]
    palette[6:9] = [0, 0, 255]
    palette[9:12] = [0, 255, 0]
    img.putpalette(palette)
    img.save(outpath)

def scheduleCutie(video_path, thumbnail_path, output_path):
    exp_name = os.path.split(os.path.splitext(video_path)[0])[1]
    print("==================")
    print(f"Scheduling actions for {exp_name}")
    print("==================")
    print("")

    exp_out_dir = os.path.abspath(output_path)
    clip_vid_base = os.path.join(exp_out_dir, "clips")
    os.makedirs(clip_vid_base, exist_ok=True)
    cutie_output_base = os.path.join(exp_out_dir, "cutie")
    os.makedirs(cutie_output_base, exist_ok=True)

    # match each segment initial frame
    print(f"Searching: {thumbnail_path}")
    segmentframes = sorted(list(filter(lambda x: ".png" in x, local_list_dir(thumbnail_path, entryType = "FILE"))))
    print("******************")
    print(f"Found segmented thumbnails: {segmentframes}")
    print("******************")
    print("")

    cutie_input_mask_base = os.path.join(exp_out_dir, "cutie_input_masks")
    common_masks_dir = os.path.join(cutie_input_mask_base, f"masks")
    common_frames_dir = os.path.join(cutie_input_mask_base, f"frames")
    os.makedirs(cutie_input_mask_base, exist_ok=True)
    os.makedirs(common_masks_dir, exist_ok=True)
    os.makedirs(common_frames_dir, exist_ok=True)

    # next, we need to get all the masks and corresponding frames...
    cap = cv2.VideoCapture(video_path)
    print(f"Opened {video_path}")
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    allmasks = sorted(segmentframes)
    for idx in range(len(allmasks)):
        curmask = allmasks[idx]
        maskframenum = int(os.path.splitext(curmask)[0])
        print(f"Copying mask {curmask} [frame = {maskframenum}]")
        cap.set(cv2.CAP_PROP_POS_FRAMES, maskframenum)
        _, frame = cap.read()
        if frame is None:
            print(f"Failed to read frame from {video_path} @ frame {maskframenum}")

        convertSegmentedToCutieIdentities(
                os.path.join(thumbnail_path, curmask),
                os.path.join(common_masks_dir, curmask))
        cv2.imwrite(os.path.join(common_frames_dir, curmask), frame)

    for sidx in range(len(segmentframes) - 1):
        forigin = segmentframes[sidx]
        rorigin = segmentframes[sidx + 1]
        forfnum = int(forigin.split(".")[0])
        rorfnum = int(rorigin.split(".")[0])
        fwdclipvid = os.path.join(clip_vid_base, f"{exp_name}.{forfnum:07d}-{rorfnum:07d}.forward.mp4")
        revclipvid = os.path.join(clip_vid_base, f"{exp_name}.{forfnum:07d}-{rorfnum:07d}.reverse.mp4")

        fwd_out_dir = os.path.join(cutie_output_base, f"{forfnum:07d}-{rorfnum:07d}.forward")
        rev_out_dir = os.path.join(cutie_output_base, f"{forfnum:07d}-{rorfnum:07d}.reverse")
        os.makedirs(fwd_out_dir, exist_ok=True)
        os.makedirs(rev_out_dir, exist_ok=True)

        clip_mask_fwddir = os.path.join(cutie_input_mask_base, f"{forfnum:07d}-{rorfnum:07d}.forward")
        clip_mask_revdir = os.path.join(cutie_input_mask_base, f"{forfnum:07d}-{rorfnum:07d}.reverse")
        os.makedirs(clip_mask_fwddir, exist_ok=True)
        os.makedirs(clip_mask_revdir, exist_ok=True)

        convertSegmentedToCutieIdentities(
                os.path.join(thumbnail_path, forigin),
                os.path.join(clip_mask_fwddir, "0000000.png"))
        clipendframenum = rorfnum - forfnum
        convertSegmentedToCutieIdentities(
                os.path.join(thumbnail_path, rorigin),
                os.path.join(clip_mask_revdir, f"0000000.png"))

        with open("inference_tasks.txt", "a") as fh:
            if os.path.exists(os.path.join(fwd_out_dir, "masks.combined")):
                print(f"Skipping {os.path.basename(fwd_out_dir)} - already complete")
            else:
                print(f"Adding task for {os.path.basename(fwd_out_dir)}")
                fh.write(f"python ../cutie_inference/cutie_inference.py --start_frame {forfnum} --end_frame {rorfnum} --video \"{os.path.abspath(video_path)}\" --common_mask_dir \"{os.path.abspath(common_masks_dir)}\" --common_frames_dir \"{os.path.abspath(common_frames_dir)}\" --clip_mask_dir \"{os.path.abspath(clip_mask_fwddir)}\" --output_dir \"{os.path.abspath(fwd_out_dir)}\" --direction=forward --clipvideo {os.path.abspath(fwdclipvid)}\n")
            if os.path.exists(os.path.join(rev_out_dir, "masks.combined")):
                print(f"Skipping {os.path.basename(rev_out_dir)} - already complete")
            else:
                print(f"Adding task for {os.path.basename(rev_out_dir)}")
                fh.write(f"python ../cutie_inference/cutie_inference.py --start_frame {forfnum} --end_frame {rorfnum} --video \"{os.path.abspath(video_path)}\" --common_mask_dir  \"{os.path.abspath(common_masks_dir)}\" --common_frames_dir  \"{os.path.abspath(common_frames_dir)}\" --clip_mask_dir  \"{os.path.abspath(clip_mask_revdir)}\" --output_dir \"{os.path.abspath(rev_out_dir)}\" --direction=reverse --clipvideo {os.path.abspath(revclipvid)}\n")
    print(f"Cutie inference tasks for {exp_name} added to inference_tasks.txt")

scheduleCutie(args.video, args.thumbnail_path, args.output_path)
