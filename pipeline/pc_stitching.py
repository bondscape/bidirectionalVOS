import cv2
import time
import os
import re
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--segments_path", required = True)
args = parser.parse_args()

segments_base = args.segments_path

if os.path.exists(os.path.join(segments_base, "post-cutie-stitching.complete")):
    print(f"DONE: {segments_base} already stitched.")
    sys.exit(0);

BATCH_SIZE = 100

def nameFromSegmentEntry(entry):
    return f"{entry['start']:07d}-{entry['end']:07d}.{entry['direction']}"

def flipIdentities(segmentation):
    return ((segmentation == 1) * 2 + (segmentation == 2) * 1).astype(np.uint8)

segments = {}
segments["forward"] = {}
segments["reverse"] = {}

def calculateIOUSingle(forward_segmentations, reverse_segmentations, id):
    # IOU calculation
    and1 = np.logical_and(forward_segmentations == id, reverse_segmentations == id)
    or1 = np.logical_or(forward_segmentations == id, reverse_segmentations == id)
    iou = np.divide(
              np.sum(and1, axis=1),
              np.sum(or1, axis=1))
    return iou

def calculateIOUs(forward_segmentations, reverse_segmentations):
    # IOU calculation
    and1 = np.logical_and(forward_segmentations == 1, reverse_segmentations == 1)
    and2 = np.logical_and(forward_segmentations == 2, reverse_segmentations == 2)
    or1 = np.logical_or(forward_segmentations == 1, reverse_segmentations == 1)
    or2 = np.logical_or(forward_segmentations == 2, reverse_segmentations == 2)
    iou = np.divide(
              (np.sum(and1, axis=1) + np.sum(and2, axis=1)),
              (np.sum(or1, axis=1) + np.sum(or2, axis=1)))
    return iou

dirs = []
for entry in os.scandir(segments_base):
    if entry.is_dir() and None != re.match("^[0-9\-]+\.", entry.name):
        region = entry.name.split(".")[0]
        direction = entry.name.split(".")[1]
        region_start = int(region.split("-")[0])
        region_end = int(region.split("-")[1])
        segments[direction][region_start] = {
                "start": region_start,
                "end": region_end,
                "direction": direction,
            }

rstarts = sorted(list(segments["forward"].keys()))
for direction in ["forward", "reverse"]:
    for ridx in range(len(rstarts)):
        rstart = rstarts[ridx]
        # stitch the segments
        if ridx == 0:
            # this segment shall be baseline
            segments[direction][rstart]["flip"] = False
            continue
        # otherwise, compare to last frame of previous segment.
        prev_segment = segments[direction][rstarts[ridx-1]]
        this_segment = segments[direction][rstarts[ridx]]
        prev_h5path = glob.glob(os.path.join(segments_base, nameFromSegmentEntry(prev_segment), "*.masks.h5"))[0]
        this_h5path = glob.glob(os.path.join(segments_base, nameFromSegmentEntry(this_segment), "*.masks.h5"))[0]

        with h5py.File(prev_h5path, "r") as h5:
            # read the "end" frame from prev_segment
            dataset = h5["masks"]
            last_final = dataset[-1]
            if prev_segment["flip"]:
                last_final = flipIdentities(last_final)

        with h5py.File(this_h5path, "r") as h5:
            # read the first frame from this segment
            dataset = h5["masks"]
            this_first = dataset[0]

        cis_score = np.sum(last_final == this_first)
        trans_score = np.sum(last_final == flipIdentities(this_first))

        if cis_score >= trans_score:
            print(f"{nameFromSegmentEntry(prev_segment)}->{nameFromSegmentEntry(this_segment)} |> Cis")
            segments[direction][rstarts[ridx]]["flip"] = False
        if trans_score > cis_score:
            print(f"{nameFromSegmentEntry(prev_segment)}->{nameFromSegmentEntry(this_segment)} |> Trans")
            segments[direction][rstarts[ridx]]["flip"] = True

first_frame = segments["forward"][rstarts[0]]["start"]
last_frame = segments["forward"][rstarts[-1]]["end"]
total_frames = last_frame + 1

comparisons = np.zeros(last_frame + 1)
alt_comparisons = np.zeros(last_frame + 1)

last_segment_start = time.time()

combined_out_created = False

for ridx in range(len(rstarts)):
    # go through each segment. flip segment if the directory says to
    # segment 0 gets no flip
    segment_start = time.time()
    print(f"Time elapsed since last segment: {segment_start - last_segment_start}")
    last_segment_start = segment_start

    absolute_fnum = rstarts[ridx]
    print(f"Combined output offset begins out: {absolute_fnum}")

    forward_seg = segments["forward"][rstarts[ridx]]
    reverse_seg = segments["reverse"][rstarts[ridx]]

    forward_h5path = glob.glob(os.path.join(segments_base, nameFromSegmentEntry(forward_seg), "*.masks.h5"))[0]
    downscaled_forward_h5path_unflipped = os.path.splitext(forward_h5path)[0] + ".downscaled.unflipped.h5"

    reverse_h5path = glob.glob(os.path.join(segments_base, nameFromSegmentEntry(reverse_seg), "*.masks.h5"))[0]
    downscaled_reverse_h5path_unflipped = os.path.splitext(reverse_h5path)[0] + ".downscaled.unflipped.h5"

    forward_h5 = h5py.File(forward_h5path, "r",
                           driver="stdio", rdcc_nbytes = 1024*1024*256)
    reverse_h5 = h5py.File(reverse_h5path, "r",
                           driver="stdio", rdcc_nbytes = 1024*1024*256)
    downscaled_unflipped_forward_h5 = h5py.File(downscaled_forward_h5path_unflipped, 'w',
                                                driver="stdio", rdcc_nbytes = 1024*1024*256)
    downscaled_unflipped_reverse_h5 = h5py.File(downscaled_reverse_h5path_unflipped, 'w',
                                                driver="stdio", rdcc_nbytes = 1024*1024*256)

    forward_dataset = forward_h5["masks"]
    reverse_dataset = reverse_h5["masks"]

    segment_frame_base = forward_seg["start"]
    chunk_size = (1, forward_dataset.shape[1], forward_dataset.shape[2])
    downscaled_chunk_size = (1, int(forward_dataset.shape[1]/4), int(forward_dataset.shape[2]/4))
    downscaled_forward_dataset_shape = (forward_dataset.shape[0],
                                        int(forward_dataset.shape[1]/4),
                                        int(forward_dataset.shape[2]/4))
    downscaled_reverse_dataset_shape = (reverse_dataset.shape[0],
                                        int(reverse_dataset.shape[1]/4),
                                        int(reverse_dataset.shape[2]/4))

    print(f"Writing chunks of size {chunk_size}")
    downscaled_unflipped_forward_dataset = downscaled_unflipped_forward_h5.create_dataset(
                                             'masks',
                                             shape=downscaled_forward_dataset_shape,
                                             chunks=downscaled_chunk_size,
                                             dtype='uint8', compression='gzip', fillvalue=255)
    downscaled_unflipped_reverse_dataset = downscaled_unflipped_reverse_h5.create_dataset(
                                             'masks',
                                             shape=downscaled_reverse_dataset_shape,
                                             chunks=downscaled_chunk_size,
                                             dtype='uint8', compression='gzip', fillvalue=255)

    if not combined_out_created:
        # open the masks_forward.h5 and masks_reverse.h5 outfiles
        out_combined_fwd_h5_path = os.path.join(segments_base, "masks_forward.h5")
        out_combined_rev_h5_path = os.path.join(segments_base, "masks_reverse.h5")
        out_combined_fwd_h5 = h5py.File(out_combined_fwd_h5_path, 'w',
                                                driver="stdio", rdcc_nbytes = 1024*1024*256)
        out_combined_rev_h5 = h5py.File(out_combined_rev_h5_path, 'w',
                                                driver="stdio", rdcc_nbytes = 1024*1024*256)
        out_combined_ds_shape = (total_frames,
                                 downscaled_forward_dataset_shape[1],
                                 downscaled_forward_dataset_shape[2])
        out_combined_ds_chunk_size = (1,
                                 downscaled_forward_dataset_shape[1],
                                 downscaled_forward_dataset_shape[2])


        out_combined_fwd_ds = out_combined_fwd_h5.create_dataset(
                                             'masks',
                                             shape=out_combined_ds_shape,
                                             chunks=out_combined_ds_chunk_size,
                                             dtype='uint8', compression='gzip', fillvalue=255)
        out_combined_rev_ds = out_combined_rev_h5.create_dataset(
                                             'masks',
                                             shape=out_combined_ds_shape,
                                             chunks=out_combined_ds_chunk_size,
                                             dtype='uint8', compression='gzip', fillvalue=255)
        combined_out_created = True


    segment_fnum = 0
    while segment_fnum < forward_dataset.shape[0]:
        n_elements = min(forward_dataset.shape[0] - segment_fnum, BATCH_SIZE)

        print(f"Working at offset {segment_fnum + segment_frame_base} for len {n_elements}")
        forward_frames = forward_dataset[segment_fnum:segment_fnum+n_elements] #.reshape(n_elements, -1)
        if forward_seg["flip"]:
            forward_frames = flipIdentities(forward_frames)
        reverse_frames = reverse_dataset[segment_fnum:segment_fnum+n_elements] #.reshape(n_elements, -1)
        if reverse_seg["flip"]:
            reverse_frames = flipIdentities(reverse_frames)

        for i in range(forward_frames.shape[0]):
            downscaled_unflipped_forward_dataset[segment_fnum+i] = forward_frames[i][1::4,1::4]
            downscaled_unflipped_reverse_dataset[segment_fnum+i] = reverse_frames[i][1::4,1::4]
            out_combined_fwd_ds[absolute_fnum+i] = forward_frames[i][1::4,1::4]
            out_combined_rev_ds[absolute_fnum+i] = reverse_frames[i][1::4,1::4]
        absolute_fnum += forward_frames.shape[0]

        iou = calculateIOUs(forward_frames.reshape(n_elements, -1),
                            reverse_frames.reshape(n_elements, -1))
        alt_iou = calculateIOUs(flipIdentities(forward_frames).reshape(n_elements, -1),
                            reverse_frames.reshape(n_elements, -1))
        comparisons[segment_frame_base+segment_fnum:segment_frame_base+segment_fnum+n_elements] = iou
        alt_comparisons[segment_frame_base+segment_fnum:segment_frame_base+segment_fnum+n_elements] = alt_iou

        segment_fnum += n_elements

    forward_h5.close()
    reverse_h5.close()
    downscaled_unflipped_forward_h5.close()
    downscaled_unflipped_reverse_h5.close()

out_combined_fwd_h5.close()
out_combined_rev_h5.close()

np.save(os.path.join(segments_base, f"iou.npy"), comparisons)
np.save(os.path.join(segments_base, f"alt_iou.npy"), alt_comparisons)
plt.plot(np.arange(len(comparisons)), comparisons, label="IOU")
plt.plot(np.arange(len(alt_comparisons)), alt_comparisons, label="ALT IOU")
plt.legend()
plt.savefig(os.path.join(segments_base, f"iou.png"))

with open(os.path.join(segments_base, "post-cutie-stitching.complete"), "w") as f:
    pass
print(f"DONE: {segments_base}")

