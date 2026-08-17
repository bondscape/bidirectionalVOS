import os
import shutil
import cv2
import colorcet as cc
from pathlib import Path
from PIL import Image
import argparse
import numpy as np
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--video",
                    help="Path to input_video.mp4",
                    required=True)
parser.add_argument("--reference_thumbnail_path",
                    help="Path to directory containing reference thumbnails",
                    required=True)
parser.add_argument("--segmented_thumbnail_path",
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

def build_palette():
    # palette for converted masks.
    # glasbey gives us distinct colors for many labels.
    palette = [0] * 768
    for label in range(1, 256):
        hex_color = cc.palette['glasbey_bw'][label - 1]

        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        palette[label*3:label*3+3] = [r,g,b]
    return palette

def convertSegmentedToCutieIdentities(inpath, outpath, ref_frame):
    # read input image - we need to detect num_planes in the image for compatibility reasons
    pil_img = Image.open(inpath)
    is_single_plane = pil_img.mode in ("P", "L")

    if is_single_plane:
        #image is already in cutie format.
        outimage = np.array(pil_img).astype(np.uint8)

    else:
        inimage = np.array(pil_img.convert("RGB"))[:, :, ::-1]  # RGB -> BGR
        if inimage.shape[2] != 3:
            print(f"Regarding {inpath}:")
            print("  Unsupported input mask format - we can handle 3-plane png, one animal per plane")
            print("  or a single 8-bit plane png with one mask for each value > 0")
            sys.exit(1)

        outimage = np.zeros(inimage.shape[:2], dtype=np.uint8)
        outimage[inimage[:, :, 0] != 0] = 1  # b channel -> label 1
        outimage[inimage[:, :, 2] != 0] = 2  # r channel -> label 2
        outimage[inimage[:, :, 1] != 0] = 3  # g channel -> label 3

    # outimage is now in the correct format
    # check to see if we need to rescale it to match thumbnail images:
    ref_image = cv2.imread(ref_frame, cv2.IMREAD_UNCHANGED)
    if ref_image is None:
        print(f"Unable to cv2.imread(ref_frame={ref_frame})!")
        sys.exit(1)

    mask_h, mask_w = outimage.shape[:2]
    ref_h, ref_w = ref_image.shape[:2]

    # aspect ratios should match exactly: mask_w/mask_h == ref_w/ref_h
    if mask_w * ref_h != mask_h * ref_w:
        print(f"Regarding mask {inpath}:")
        print(f"Regarding ref {ref_frame}:")
        print(f"  ERROR: mask/ref aspect ratio mismatch")
        print(f"  mask size: {mask_w}x{mask_h}, ref size: {ref_w}x{ref_h}")
        sys.exit(1)

    outimage = cv2.resize(outimage, (ref_w, ref_h),
                          interpolation=cv2.INTER_NEAREST_EXACT)

    img = Image.fromarray(outimage, mode='L').convert('P')
    img.putpalette(build_palette())
    img.save(outpath)

def scheduleCutie(video_path, reference_thumbnail_path, segmented_thumbnail_path, output_path):
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

    total_frame_count = get_video_frame_count(video_path)
    print(f"{video_path} is {total_frame_count} frames.")

    # match each segment initial frame
    print(f"Searching: {segmented_thumbnail_path}")
    segmentframes = sorted(list(filter(lambda x: ".png" in x,
                           local_list_dir(segmented_thumbnail_path, entryType = "FILE"))))

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
    allmasks = sorted(segmentframes)
    for idx in range(len(allmasks)):
        curmask = allmasks[idx]
        maskframenum = int(os.path.splitext(curmask)[0])
        print(f"Copying mask {curmask} [frame = {maskframenum}]")
        ref_frame_path = os.path.join(reference_thumbnail_path, f"{maskframenum:07d}.jpg")
        if not os.path.exists(ref_frame_path):
            print(f"Failed to read frame {ref_frame_path}")
            die

        convertSegmentedToCutieIdentities(
                os.path.join(segmented_thumbnail_path, curmask),
                os.path.join(common_masks_dir, curmask),
                ref_frame_path)
        shutil.copyfile(ref_frame_path, os.path.join(common_frames_dir, curmask))

    for sidx in range(len(segmentframes) - 1):
        forigin = segmentframes[sidx]
        rorigin = segmentframes[sidx + 1]
        forfnum = int(forigin.split(".")[0])
        rorfnum = int(rorigin.split(".")[0])

        forf_ref_frame_path = os.path.join(reference_thumbnail_path, f"{forfnum:07d}.jpg")
        rorf_ref_frame_path = os.path.join(reference_thumbnail_path, f"{forfnum:07d}.jpg")

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
                os.path.join(segmented_thumbnail_path, forigin),
                os.path.join(clip_mask_fwddir, "0000000.png"),
                forf_ref_frame_path)
        clipendframenum = rorfnum - forfnum
        convertSegmentedToCutieIdentities(
                os.path.join(segmented_thumbnail_path, rorigin),
                os.path.join(clip_mask_revdir, f"0000000.png"),
                rorf_ref_frame_path)

        with open("inference_tasks.txt", "a") as fh:
            if os.path.exists(os.path.join(fwd_out_dir, "masks.combined")):
                print(f"Skipping {os.path.basename(fwd_out_dir)} - already complete")
            else:
                print(f"Adding task for {os.path.basename(fwd_out_dir)}")
                fh.write(f"python ../cutie_inference/cutie_inference.py --start_frame {forfnum} --end_frame {rorfnum} --video \"{os.path.abspath(video_path)}\" --common_frames_dir \"{os.path.abspath(common_frames_dir)}\" --clip_mask_dir \"{os.path.abspath(clip_mask_fwddir)}\" --output_dir \"{os.path.abspath(fwd_out_dir)}\" --direction=forward --clipvideo {os.path.abspath(fwdclipvid)}\n")

            if os.path.exists(os.path.join(rev_out_dir, "masks.combined")):
                print(f"Skipping {os.path.basename(rev_out_dir)} - already complete")
            else:
                print(f"Adding task for {os.path.basename(rev_out_dir)}")
                fh.write(f"python ../cutie_inference/cutie_inference.py --start_frame {forfnum} --end_frame {rorfnum} --video \"{os.path.abspath(video_path)}\" --common_frames_dir  \"{os.path.abspath(common_frames_dir)}\" --clip_mask_dir  \"{os.path.abspath(clip_mask_revdir)}\" --output_dir \"{os.path.abspath(rev_out_dir)}\" --direction=reverse --clipvideo {os.path.abspath(revclipvid)}\n")
    print(f"Cutie inference tasks for {exp_name} added to inference_tasks.txt")

scheduleCutie(args.video, args.reference_thumbnail_path, args.segmented_thumbnail_path, args.output_path)
