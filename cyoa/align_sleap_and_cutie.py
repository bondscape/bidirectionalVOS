import cv2
import os
import numpy as np
import pandas as pd
import h5py
import glob
import matplotlib.pyplot as plt
#import scipy.signal
import time
import json
import pathlib
import sys
import shutil

import argparse

font = cv2.FONT_HERSHEY_TRIPLEX
font_size = 0.70
font_thickness = 2
font_backing = False
font_main_color = (140, 30, 70)

def loadPartsConfig(parts_config_path):
    if not os.path.exists(parts_config_path):
        print(f"Couldn't find parts.config {parts_config_path}")
        sys.exit(1)
    with open(parts_config_path, "r") as fh:
        parts_config = json.loads(fh.read())
        return parts_config

def getAnimalIdentities(identities_config_path):
    with open(identities_config_path, "r") as fh:
        identities = json.loads(fh.read())
        return identities

def group_indices(indices):
    if len(indices) == 0:
        return []
    indices = np.sort(np.unique(indices))
    split_points = np.where(np.diff(indices) != 1)[0] + 1
    groups = np.split(indices, split_points)

    result = []
    for group in groups:
        if len(group) == 1:
            result.append(f"{group[0]}")
        else:
            result.append(f"{group[0]}-{group[-1]}")
    return result

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", help="path to video.mp4", required=True)
parser.add_argument("--annotations", help="path video.annotations.csv", required=True)
parser.add_argument("--identities_config_path", help="path to video.mp4", required=True)
parser.add_argument("--cutie_path", help="path to cutie segments folder", required=True)
parser.add_argument("--sleap_path", help="path to sleap tracked.cleaned.h5 file", required=True)
parser.add_argument("--sleap_remaster_path", default="", help="path to sleap remaster h5 file", required=False)
parser.add_argument("--parts_config", help="path to parts.config file", required=True)
parser.add_argument("--output_path", help="output path (directory)", required=True)
parser.add_argument("--do_video", action="store_true")

args = parser.parse_args()

exp = os.path.splitext(os.path.basename(args.video_path))[0]
exp_with_ext = os.path.basename(args.video_path)

dataframe_for_results = f"{exp}.annotations.csv"
do_video = args.do_video

node_id_to_node_name = {}
node_name_to_node_id = {}
valid_node_names = []
valid_node_ids = []
n_valid_node_ids = 0

parts_config = loadPartsConfig(args.parts_config)
valid_node_names = parts_config["valid_node_names"]

anSexes = getAnimalIdentities(args.identities_config_path)["identities"]

CHUNK_LENGTH = 512


def load_zod_csv(csvpath):
    df = pd.DataFrame(columns = ["region_start", "region_end", "source", "previously_scored", "cutie_forward_quality", "cutie_forward_swapped", "cutie_reverse_quality", "cutie_reverse_swapped", "best_cutie", "best_cutie_swapped", "best_cutie_quality", "sleap_raw_quality", "sleap_raw_swapped", "sleap_interpolated_quality"])
    if not os.path.exists(csvpath):
        print(f"Unable to locate {csvpath}")
        sys.exit(1)

    df = pd.read_csv(csvpath, keep_default_na=False)
    if "previously_scored" not in df.columns:
        df["previously_scored"] = np.ones(len(df))
    if "source" not in df.columns:
        df["source"] = ["UNKNOWN"] * len(df)

    df["region_start"] = df["region_start"].astype(int)
    df["region_end"] = df["region_end"].astype(int)
    df["previously_scored"] = df["previously_scored"].astype(int)

    return df

def resolveAnnotations(frame_count, csvpath):
    zod_df = load_zod_csv(csvpath)
    zod_df = zod_df.sort_values("region_start")

    cutie_mappings = {
        # should be modified by sex mappings before initiating zod resolution
        "cutie_fwd_0_identity": np.zeros(frame_count, dtype=float),
        "cutie_fwd_1_identity": np.ones(frame_count, dtype=float),
        "cutie_rev_0_identity": np.zeros(frame_count, dtype=float),
        "cutie_rev_1_identity": np.ones(frame_count, dtype=float),

        "cutie_fwd_valid": np.ones(frame_count, dtype=int),
        "cutie_rev_valid": np.ones(frame_count, dtype=int),

        # -1 = indifferent/invalid
        "cutie_preferred": -1 * np.ones(frame_count, dtype=int),

        # sleap data is assumed to be usable (unless excluded by ZOBs)
        "sleap_raw_usable": np.ones(frame_count, dtype=int),
        "sleap_interpolate": np.zeros(frame_count, dtype=int),
        "sleap_interpolate_swap": np.zeros(frame_count, dtype=int),
        "sleap_remaster_infill": np.zeros(frame_count, dtype=int),
    }

    zod_idx = 0
    for zod_idx in range(len(zod_df)):
        r_start = zod_df.loc[zod_idx, 'region_start']
        r_end = zod_df.loc[zod_idx, 'region_end']
        c_fwd_swapped = zod_df.loc[zod_idx, 'cutie_forward_swapped'] == "SWAPPED"
        c_rev_swapped = zod_df.loc[zod_idx, 'cutie_reverse_swapped'] == "SWAPPED"
        c_fwd_quality = zod_df.loc[zod_idx, 'cutie_forward_quality']
        c_rev_quality = zod_df.loc[zod_idx, 'cutie_reverse_quality']
        c_best = zod_df.loc[zod_idx, 'best_cutie']
        slp_raw_quality = zod_df.loc[zod_idx, 'sleap_raw_quality']
        slp_interpolate_quality = zod_df.loc[zod_idx, 'sleap_interpolated_quality']
        slp_interpolate_swapped = zod_df.loc[zod_idx, 'sleap_raw_swapped'] == "SWAPPED"

        if c_fwd_swapped:
            # invalidate fwd
            cutie_mappings["cutie_fwd_valid"][r_start:r_end+1] = 0
            cutie_mappings["cutie_fwd_0_identity"][r_start:r_end+1] = np.nan
            cutie_mappings["cutie_fwd_1_identity"][r_start:r_end+1] = np.nan
            tmp = cutie_mappings["cutie_fwd_0_identity"][r_end+1:].copy()
            cutie_mappings["cutie_fwd_0_identity"][r_end+1:] = cutie_mappings["cutie_fwd_1_identity"][r_end+1:]
            cutie_mappings["cutie_fwd_1_identity"][r_end+1:] = tmp
        if c_rev_swapped:
            # invalidate rev
            cutie_mappings["cutie_rev_valid"][r_start:r_end+1] = 0
            cutie_mappings["cutie_rev_0_identity"][r_start:r_end+1] = np.nan
            cutie_mappings["cutie_rev_1_identity"][r_start:r_end+1] = np.nan
            tmp = cutie_mappings["cutie_rev_0_identity"][r_end+1:].copy()
            cutie_mappings["cutie_rev_0_identity"][r_end+1:] = cutie_mappings["cutie_rev_1_identity"][r_end+1:]
            cutie_mappings["cutie_rev_1_identity"][r_end+1:] = tmp
        if c_fwd_quality == "GARBAGE":
            cutie_mappings["cutie_fwd_valid"][r_start:r_end+1] = 0
        if c_rev_quality == "GARBAGE":
            cutie_mappings["cutie_rev_valid"][r_start:r_end+1] = 0

        if c_best == "FORWARD":
            cutie_mappings["cutie_preferred"][r_start:r_end+1] = 0
        if c_best == "REVERSE":
            cutie_mappings["cutie_preferred"][r_start:r_end+1] = 1
        if c_best in ["NONE", ""]:
            cutie_mappings["sleap_remaster_infill"][r_start:r_end+1] = 1

        if slp_raw_quality in ["GOOD", "GAUCHE"]:
            cutie_mappings["sleap_raw_usable"][r_start:r_end+1] = 1

        if slp_raw_quality == "GAUCHE" and slp_interpolate_quality == "GOOD":
            cutie_mappings["sleap_raw_usable"][r_start:r_end+1] = 0

        if slp_interpolate_quality in ["GOOD", "GAUCHE"] and cutie_mappings["sleap_raw_usable"][r_start] == 0:
            cutie_mappings["sleap_interpolate"][r_start:r_end+1] = 1
            cutie_mappings["sleap_interpolate_swap"][r_start:r_end+1] = slp_interpolate_swapped

    return zod_df, cutie_mappings

def getSleapData(sleap_h5_path):
    global node_id_to_node_name
    global node_node_to_node_id
    global valid_node_names
    global valid_node_ids
    global n_valid_node_ids

    print(f"loading {sleap_h5_path}")
    with h5py.File(sleap_h5_path, "r") as f:
        data = {
            "dset_names": list(f.keys()),
            "locations": f["tracks"][:].T,
            "instance_scores": f["instance_scores"][:].T,
            "tracking_scores": f["tracking_scores"][:].T,
            "point_scores": f["point_scores"][:].T,
            "node_names": [n.decode() for n in f["node_names"][:]]
        }

    node_id_to_node_name = {}
    node_name_to_node_id = {}

    for nid in range(len(data["node_names"])):
        nn = data["node_names"][nid]
        node_id_to_node_name[nid] = nn
        node_name_to_node_id[nn] = nid

    valid_node_ids = list(map(lambda x: node_name_to_node_id[x], valid_node_names))
    n_valid_node_ids = len(valid_node_ids)

    return data

filters = {
    5: np.ones(5*5).reshape(5,5),
}

def dilate(image, px, thresh = 1):
    if image.shape[2] == 1:
        # dilate() returns [h, w, 1] as [h, w], so we need to reexpand that dimension.
        print(f"Expanding dims on {image.shape}")
        return np.expand_dims(cv2.dilate(image, filters[px]), axis=2)
    return cv2.dilate(image, filters[px].astype(int))

# Reset only
frame_tracking = {}

plt.rcParams['figure.figsize'] = [12, 9]

def getH5(dir, downscaled = True):
    if downscaled:
        return glob.glob(os.path.join(dir, "*.masks.downscaled.unflipped.h5"))[0]
    return glob.glob(os.path.join(dir, "*.masks.unflipped.h5"))[0]

def sleap_to_consensus_cutie_mapping(sleap_data, cutie_masks,
                        dilation_kernel = 5, downscaled = True):
    sleap_raw = sleap_data["raw"]["locations"]
    length = min(sleap_data["raw"]["locations"].shape[0], cutie_masks["consensus_masks_ds"].shape[0])

    def pt_identity_xform(x):
        return int(x)
    def pt_downscale_xform(x):
        return int(x * scale_factor)
    def coord_identity_xform(xs):
        return np.round(xs).astype(int)
    def coord_downscale_xform(xs):
        return np.floor(xs/4).astype(int)

    scale_factor = 1
    coord_xform = coord_identity_xform
    pt_xform = pt_identity_xform
    if downscaled:
        pt_xform = pt_downscale_xform
        coord_xform = coord_downscale_xform
        scale_factor = 0.25

    sleap_nodes_to_cutie_masks = np.zeros((length, 2, sleap_raw.shape[1], 2))

    curidx = 0
    while curidx < length:
        thislength = min(length-curidx, CHUNK_LENGTH)
        print(f"Processing sleap<=>cutie mappings. On {curidx} / {length}. chunklen: {thislength}")

        cutie_data = cutie_masks["consensus_masks_ds"][curidx:curidx+thislength]

        # transposed in order to treat each image as a separate plane with cv2.
        cutie_data = np.transpose(cutie_data, axes = [1,2,0])

        # dilate and separate cutie masks; {tracks, y, x, frames}
        dilated_masks = np.zeros((2, cutie_data.shape[0], cutie_data.shape[1], cutie_data.shape[2]))

        dilated_masks[0,:,:,:] = dilate((cutie_data == 1).astype(np.uint8), dilation_kernel)
        dilated_masks[1,:,:,:] = dilate((cutie_data == 2).astype(np.uint8), dilation_kernel)

        del cutie_data

        for slpid in [0,1]:
            for cutieid in [0,1]:
                for anpid in range(sleap_raw.shape[1]):
                    correspondences = np.zeros(thislength, dtype=np.float64)
                    correspondences[:] = np.nan

                    for chunk_sub_idx in range(thislength):
                        xcoord = sleap_raw[curidx+chunk_sub_idx, anpid, 0, slpid]
                        ycoord = sleap_raw[curidx+chunk_sub_idx, anpid, 1, slpid]
                        if np.isnan(xcoord):
                            continue
                        correspondences[chunk_sub_idx] = dilated_masks[cutieid,
                                                                       coord_xform(ycoord),
                                                                       coord_xform(xcoord),
                                                                       chunk_sub_idx] > 0
                    sleap_nodes_to_cutie_masks[curidx:curidx+thislength, slpid, anpid, cutieid] = correspondences
        curidx += CHUNK_LENGTH
    return sleap_nodes_to_cutie_masks

def export_consensus_cutie(
    consensus_h5_path,
    rejected_h5_path,
    start, length,
    cutie_masks,
    downscaled = True):

    if os.path.exists(consensus_h5_path) and os.path.exists(rejected_h5_path):
        print("Already exported consensus cutie.")
        return

    print(f"Exporting best cutie masks!")
    export_cutie_fh = h5py.File(consensus_h5_path + ".tmp.h5", "w", driver="stdio", rdcc_nbytes=1024*1024*256)
    export_cutie_ds = export_cutie_fh.create_dataset(
                            'masks',
                            shape=cutie_masks["masks_fwd_ds"].shape,
                            chunks=(1, cutie_masks["masks_fwd_ds"].shape[1], cutie_masks["masks_fwd_ds"].shape[2]),
                            dtype="uint8",
                            compression="gzip")
    export_rejected_cutie_fh = h5py.File(rejected_h5_path + ".tmp.h5", "w", driver="stdio", rdcc_nbytes=1024*1024*256)
    export_rejected_cutie_ds = export_rejected_cutie_fh.create_dataset(
                            'masks',
                            shape=cutie_masks["masks_fwd_ds"].shape,
                            chunks=(1, cutie_masks["masks_fwd_ds"].shape[1], cutie_masks["masks_fwd_ds"].shape[2]),
                            dtype="uint8",
                            compression="gzip")

    def pt_identity_xform(x):
        return int(x)
    def pt_downscale_xform(x):
        return int(x * scale_factor)
    def coord_identity_xform(xs):
        return np.round(xs).astype(int)
    def coord_downscale_xform(xs):
        return np.round(xs/4).astype(int)

    scale_factor = 1
    coord_xform = coord_identity_xform
    pt_xform = pt_identity_xform
    if downscaled:
        pt_xform = pt_downscale_xform
        coord_xform = coord_downscale_xform
        scale_factor = 0.25

    # align to sexes
    def getAnCxy(mask, id):
        ypos, xpos = np.nonzero(mask == id)
        return np.mean(xpos)/scale_factor, np.mean(ypos)/scale_factor

    def mapSexes(annot_F, annot_M, c_cutie_an0, c_cutie_an1):
        d0F = ((c_cutie_an0[0] - annot_F[0])**2 + (c_cutie_an0[1] - annot_F[1])**2) ** 0.5
        d0M = ((c_cutie_an0[0] - annot_M[0])**2 + (c_cutie_an0[1] - annot_M[1])**2) ** 0.5
        d1F = ((c_cutie_an1[0] - annot_F[0])**2 + (c_cutie_an1[1] - annot_F[1])**2) ** 0.5
        d1M = ((c_cutie_an1[0] - annot_M[0])**2 + (c_cutie_an1[1] - annot_M[1])**2) ** 0.5
        print(d0F)
        print(d0M)
        print(d1F)
        print(d1M)
        if d0F < d0M and d1M < d1F:
            return ["FEMALE", "MALE"]
        elif d0M < d0F and d1F < d1M:
            return ["MALE", "FEMALE"]
        else:
            print("Unforeseen mapping issue!")
            return ["MALE", "FEMALE"]
            sys.exit(1)

    cfwd0 = getAnCxy(cutie_masks["masks_fwd_ds"][0], 1)
    cfwd1 = getAnCxy(cutie_masks["masks_fwd_ds"][0], 2)
    crev0 = getAnCxy(cutie_masks["masks_rev_ds"][0], 1)
    crev1 = getAnCxy(cutie_masks["masks_rev_ds"][0], 2)
    print(anSexes)
    cfwd_sexmap = mapSexes(anSexes["female"], anSexes["male"], cfwd0, cfwd1)
    crev_sexmap = mapSexes(anSexes["female"], anSexes["male"], crev0, crev1)

    # Note: output track 1 = female, 2 = male
    export_cutie_sex = {
        "FEMALE": 1,
        "MALE": 2,
    }

    chunkbase = start
    while chunkbase < start + length:
        thislength = min(start+length-chunkbase, CHUNK_LENGTH)

        cutie_data_fwd = cutie_masks["masks_fwd_ds"][chunkbase:chunkbase+thislength]
        cutie_data_rev = cutie_masks["masks_rev_ds"][chunkbase:chunkbase+thislength]
        chunk_export_masks = np.zeros_like(cutie_data_fwd)
        chunk_export_rejected_masks = np.zeros_like(cutie_data_fwd)

        framenum = chunkbase
        chunkidx = 0
        while chunkidx < thislength:
            cutie_choice = ""
            output = np.zeros_like(cutie_data_fwd[0])
            reject = np.zeros_like(cutie_data_fwd[0])
            if cutie_animal_mappings["cutie_fwd_valid"][framenum] or cutie_animal_mappings["cutie_rev_valid"][framenum]:
                # which cutie set are we using for this frame?
                if cutie_animal_mappings["cutie_preferred"][framenum] in [-1, 0]:
                    cutie_choice = "FORWARD"
                elif cutie_animal_mappings["cutie_preferred"][framenum] == 1:
                    cutie_choice = "REVERSE"
                else:
                    print(f"Unexpected cutie choice at frame {i}")
                    sys.exit(1)

                # given our choice:
                if cutie_choice == "FORWARD":
                    # what this does:
                    #   for cutie input track value = 1, look up animal-id as tracked by ZOBs.
                    #   look up that animal-id in the sexmap (is that male or female?
                    #   get the cutie export track value (1 = Female, 2 = Male) for associated sex.

                    # prime track
                    c0_aid = cutie_animal_mappings["cutie_fwd_0_identity"][framenum]
                    c1_aid = cutie_animal_mappings["cutie_fwd_1_identity"][framenum]
                    if not np.isnan(c0_aid):
                        c0_export_track = export_cutie_sex[cfwd_sexmap[int(c0_aid)]]
                        output += (cutie_data_fwd[chunkidx, :, :] == 1).astype(np.uint8) * c0_export_track
                    if not np.isnan(c1_aid):
                        c1_export_track = export_cutie_sex[cfwd_sexmap[int(c1_aid)]]
                        output += (cutie_data_fwd[chunkidx, :, :] == 2).astype(np.uint8) * c1_export_track

                    # rejected track
                    c0_aid = cutie_animal_mappings["cutie_rev_0_identity"][framenum]
                    c1_aid = cutie_animal_mappings["cutie_rev_1_identity"][framenum]
                    if not np.isnan(c0_aid):
                        c0_export_track = export_cutie_sex[crev_sexmap[int(c0_aid)]]
                        reject += (cutie_data_rev[chunkidx, :, :] == 1).astype(np.uint8) * c0_export_track
                    if not np.isnan(c1_aid):
                        c1_export_track = export_cutie_sex[crev_sexmap[int(c1_aid)]]
                        reject += (cutie_data_rev[chunkidx, :, :] == 2).astype(np.uint8) * c1_export_track
                if cutie_choice == "REVERSE":
                    # prime track
                    c0_aid = cutie_animal_mappings["cutie_rev_0_identity"][framenum]
                    c1_aid = cutie_animal_mappings["cutie_rev_1_identity"][framenum]
                    if not np.isnan(c0_aid):
                        c0_export_track = export_cutie_sex[crev_sexmap[int(c0_aid)]]
                        output += (cutie_data_rev[chunkidx, :, :] == 1).astype(np.uint8) * c0_export_track
                    if not np.isnan(c1_aid):
                        c1_export_track = export_cutie_sex[crev_sexmap[int(c1_aid)]]
                        output += (cutie_data_rev[chunkidx, :, :] == 2).astype(np.uint8) * c1_export_track

                    # rejected track
                    c0_aid = cutie_animal_mappings["cutie_fwd_0_identity"][framenum]
                    c1_aid = cutie_animal_mappings["cutie_fwd_1_identity"][framenum]
                    if not np.isnan(c0_aid):
                        c0_export_track = export_cutie_sex[cfwd_sexmap[int(c0_aid)]]
                        reject += (cutie_data_fwd[chunkidx, :, :] == 1).astype(np.uint8) * c0_export_track
                    if not np.isnan(c1_aid):
                        c1_export_track = export_cutie_sex[cfwd_sexmap[int(c1_aid)]]
                        reject += (cutie_data_fwd[chunkidx, :, :] == 2).astype(np.uint8) * c1_export_track

            chunk_export_masks[chunkidx] = output
            chunk_export_rejected_masks[chunkidx] = reject

            framenum += 1
            chunkidx += 1
        # write these chunks
        print(f"exported consensus cutie @ {chunkbase} + {thislength}")
        export_cutie_ds[chunkbase:chunkbase+thislength] = chunk_export_masks
        export_rejected_cutie_ds[chunkbase:chunkbase+thislength] = chunk_export_rejected_masks
        chunkbase += thislength
    export_cutie_fh.close()
    export_rejected_cutie_fh.close()
    print("Exported consensus cutie!")
    shutil.move(consensus_h5_path + ".tmp.h5", consensus_h5_path)
    shutil.move(rejected_h5_path + ".tmp.h5", rejected_h5_path)

# figure out all the infill stats
def align_sleap(sleap_data,
                sleap_save_path,
                cutie_masks,
                cutie_animal_mappings,
                sleap_to_consensus_cutie_masks):

    cutie_display_key = "consensus_masks_ds"
    cutie_alignment_key = "consensus_masks_ds"

    sleap_raw = sleap_data["raw"]["locations"]
    sleap_remaster = None
    if sleap_data['remaster'] != None:
        sleap_remaster = sleap_data["remaster"]["locations"]

    slp_anid0_cutie_0_pts = np.sum(sleap_to_consensus_cutie_masks[:, 0, valid_node_ids, 0] == 1, axis=1)
    slp_anid0_cutie_1_pts = np.sum(sleap_to_consensus_cutie_masks[:, 0, valid_node_ids, 1] == 1, axis=1)
    slp_anid1_cutie_0_pts = np.sum(sleap_to_consensus_cutie_masks[:, 1, valid_node_ids, 0] == 1, axis=1)
    slp_anid1_cutie_1_pts = np.sum(sleap_to_consensus_cutie_masks[:, 1, valid_node_ids, 1] == 1, axis=1)

    slp_anid0_valid_pts = np.sum(~np.isnan(sleap_to_consensus_cutie_masks[:, 0, valid_node_ids, 0]), axis=1)
    slp_anid1_valid_pts = np.sum(~np.isnan(sleap_to_consensus_cutie_masks[:, 1, valid_node_ids, 0]), axis=1)

    accuracy_s0_c0 = np.divide(slp_anid0_cutie_0_pts, slp_anid0_valid_pts, where=slp_anid0_valid_pts>0)
    accuracy_s0_c1 = np.divide(slp_anid0_cutie_1_pts, slp_anid0_valid_pts, where=slp_anid0_valid_pts>0)
    accuracy_s1_c0 = np.divide(slp_anid1_cutie_0_pts, slp_anid1_valid_pts, where=slp_anid1_valid_pts>0)
    accuracy_s1_c1 = np.divide(slp_anid1_cutie_1_pts, slp_anid1_valid_pts, where=slp_anid1_valid_pts>0)

    accuracy = np.stack([accuracy_s0_c0, accuracy_s0_c1, accuracy_s1_c0, accuracy_s1_c1]).T

    # track [...., 0:4] = cutie-0, cutie-1, unmappable A, unmappable B
    sleap_with_cutie_ident = np.full((sleap_to_consensus_cutie_masks.shape[0], 9, 2, 4), np.nan)

    # score = accuracy * pts
    cisConsensusScore = slp_anid0_cutie_0_pts*accuracy_s0_c0 + slp_anid1_cutie_1_pts*accuracy_s1_c1
    transConsensusScore = slp_anid0_cutie_1_pts*accuracy_s0_c1 + slp_anid1_cutie_0_pts*accuracy_s1_c0

    prev_state = "NONE"
    numSwaps = 0
    localized_swaps = []
    post_gap_swaps = []

    infill_absent_cutie = []
    infill_absent_sleap = []
    infill_non_overlap = []
    infill_nontrivial = []
    infill_nontrivial_matchdata = {}
    infill_performed = []

    for i in range(accuracy.shape[0]):
        this_frame_is_gap = False
        if cisConsensusScore[i] > transConsensusScore[i]:
            sleap_with_cutie_ident[i, :, :, 0:2] = sleap_raw[i, valid_node_ids, :, :]
            new_state = "CIS"
        elif transConsensusScore[i] > cisConsensusScore[i]:
            sleap_with_cutie_ident[i, :, :, 0] = sleap_raw[i, valid_node_ids, :, 1]
            sleap_with_cutie_ident[i, :, :, 1] = sleap_raw[i, valid_node_ids, :, 0]
            new_state = "TRANS"
        else:
            if slp_anid0_cutie_0_pts[i] == 0 and slp_anid0_cutie_1_pts[i] == 0 and \
                   slp_anid1_cutie_0_pts[i] == 0 and slp_anid1_cutie_1_pts[i] == 0:
                # trivial unalignable scenario: broken down by cause:
                if np.sum(cutie_masks[cutie_alignment_key][i]) == 0:
                    # absent cutie
                    infill_absent_cutie.append(i)
                elif np.sum(~np.isnan(sleap_data["raw"]["locations"][i,:,:,:].reshape(-1))) == 0:
                    infill_absent_sleap.append(i)
                else:
                    infill_non_overlap.append(i)
                this_frame_is_gap = True
            else:
                # No cis vs trans status available for {i} - despite some valid points!
                infill_nontrivial.append(i)
                infill_nontrivial_matchdata[i] = {
                    "s0c0": slp_anid0_cutie_0_pts[i],
                    "s0c1": slp_anid0_cutie_1_pts[i],
                    "s1c0": slp_anid1_cutie_0_pts[i],
                    "s1c1": slp_anid1_cutie_1_pts[i],
                    "cis": np.round(cisConsensusScore[i], 2),
                    "trans": np.round(transConsensusScore[i], 2),
                    "cis_a0c0_pts": slp_anid0_cutie_0_pts[i],
                    "cis_a0c0_acc": accuracy_s0_c0[i],
                    "cis_a1c1_pts": slp_anid1_cutie_1_pts[i],
                    "cis_a1c1_acc": accuracy_s1_c1[i],
                    "trans_a0c0_pts": slp_anid0_cutie_1_pts[i],
                    "trans_a0c0_acc": accuracy_s0_c1[i],
                    "trans_a1c1_pts": slp_anid1_cutie_0_pts[i],
                    "trans_a1c1_acc": accuracy_s1_c0[i],
                }
            sleap_with_cutie_ident[i, :, :, 2] = sleap_raw[i, valid_node_ids, :, 0]
            sleap_with_cutie_ident[i, :, :, 3] = sleap_raw[i, valid_node_ids, :, 1]

        if prev_state == "CIS" and new_state == "TRANS":
            if last_frame_was_gap == True:
                post_gap_swaps.append(i)
            else:
                localized_swaps.append(i)
            numSwaps+=1
        elif prev_state == "TRANS" and new_state == "CIS":
            if last_frame_was_gap == True:
                post_gap_swaps.append(i)
            else:
                localized_swaps.append(i)
            numSwaps+=1
        prev_state = new_state
        last_frame_was_gap = this_frame_is_gap

    print(f"After {accuracy.shape[0]}: {numSwaps} swaps:")
    print(f"\tLocalized Swaps: {localized_swaps} [n={len(localized_swaps)}]")
    print(f"\tPost-Gap Swaps: {post_gap_swaps} [n={len(post_gap_swaps)}]")

    # pass 2 - perform infills
    def dist2d(a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    for framenum in range(accuracy.shape[0]):
        # pull in sleap points from infill, if appropriate
        if cutie_animal_mappings["sleap_remaster_infill"][framenum] == 1:
            if cutie_animal_mappings["sleap_remaster_infill"][framenum-1] == 0:
                # this is the transition point from alignable sleap data to
                # sleap-data that has been manually aligned.
                # associate sleap tracks with their counterparts in the prior frame, then
                # this assignment is safe to keep until next regular->infill transition.
                prev_s0_xy = np.nanmean(sleap_with_cutie_ident[framenum-1, :, :, 0], axis=0)
                prev_s1_xy = np.nanmean(sleap_with_cutie_ident[framenum-1, :, :, 1], axis=0)
                infill_s0_xy = np.nanmean(sleap_remaster[framenum-1, :, :, 0], axis=0)
                infill_s1_xy = np.nanmean(sleap_remaster[framenum-1, :, :, 1], axis=0)

                cisScore = dist2d(prev_s0_xy, infill_s0_xy) + dist2d(prev_s1_xy, infill_s1_xy)
                transScore = dist2d(prev_s0_xy, infill_s1_xy) + dist2d(prev_s1_xy, infill_s0_xy)
                if cisScore < transScore:
                    infill_mode = "CIS"
                    print(f"Initiating CIS infill from remastered segment @ {framenum}")
                else:
                    infill_mode = "TRANS"
                    print(f"Initiating TRANS infill fromm remastered segment @ {framenum}")
            if infill_mode == "CIS":
                sleap_with_cutie_ident[framenum, :, :, 0] = sleap_remaster[framenum, valid_node_ids, :, 0]
                sleap_with_cutie_ident[framenum, :, :, 1] = sleap_remaster[framenum, valid_node_ids, :, 1]
                infill_performed.append(framenum)
            elif infill_mode == "TRANS":
                sleap_with_cutie_ident[framenum, :, :, 0] = sleap_remaster[framenum, valid_node_ids, :, 1]
                sleap_with_cutie_ident[framenum, :, :, 1] = sleap_remaster[framenum, valid_node_ids, :, 0]
                infill_performed.append(framenum)
            else:
                print("unexpected infill mode: {infill_mode}")
                sys.exit(1)

    if sleap_save_path != "" and sleap_save_path != None:
        print("Saving sleap data to {sleap_save_path}")
        np.save(sleap_save_path, sleap_with_cutie_ident)

    print(f"Trivial infills were desired:")
    print(f"\tAbsent cutie: {group_indices(infill_absent_cutie)} [n={len(infill_absent_cutie)}]")
    print(f"\tAbsent SLEAP: {group_indices(infill_absent_sleap)} [n={len(infill_absent_sleap)}]")
    print(f"\tNon-overlapping: {group_indices(infill_non_overlap)} [n={len(infill_non_overlap)}]")

    print(f"Non-trivial infills were desired:")
    print(f"\tFrames:{group_indices(infill_nontrivial)}")
    print(f"\tDetails:")
    for fid in infill_nontrivial:
        print(f"\t\tFrame: {fid} - {infill_nontrivial_matchdata[fid]}")

    print(f"Infills performed on: {group_indices(infill_performed)} [n={len(infill_performed)}]")

    print(f"Frames with unperformed infills:")
    remainder_absent_cutie = list(set(infill_absent_cutie) - set(infill_performed))
    remainder_absent_sleap = list(set(infill_absent_sleap) - set(infill_performed))
    remainder_non_overlap = list(set(infill_non_overlap) - set(infill_performed))
    remainder_nontrivial = list(set(infill_nontrivial) - set(infill_performed))

    print(f"\tRemainder absent cutie: {group_indices(remainder_absent_cutie)} [n={len(remainder_absent_cutie)}]")
    print(f"\tRemainder absent SLEAP: {group_indices(remainder_absent_sleap)} [n={len(remainder_absent_sleap)}]")
    print(f"\tRemainder non-overlap: {group_indices(remainder_non_overlap)} [n={len(remainder_non_overlap)}]")
    print(f"\tRemainder nontrivial: {group_indices(remainder_nontrivial)} [n={len(remainder_nontrivial)}]")
    for fid in remainder_nontrivial:
        print(f"\t\tFrame: {fid} - {infill_nontrivial_matchdata[fid]}")

    return {
        "sleap_with_cutie_ident": sleap_with_cutie_ident,
        "infill_performed": infill_performed,
        "remainder_absent_cutie": remainder_absent_cutie,
        "remainder_absent_sleap": remainder_absent_sleap,
        "remainder_non_overlap": remainder_non_overlap,
        "remainder_nontrivial": remainder_nontrivial,
    }

def draw_section(input_mp4, output_mp4,
                 start, length,
                 sleap_data,
                 sleap_with_cutie_ident,
                 cutie_masks,
                 cutie_animal_mappings,
                 sleap_to_consensus_cutie_masks,
                 options,
                 downscaled = True):

    cutie_alignment_key = "consensus_masks_ds"

    if "rejected_cutie_track" in options["options"]:
        print(f"Showing aligned sleap relative to dispreferred cutie")
        cutie_display_key = "rejected_masks_ds"
    else:
        print(f"Showing aligned sleap relative to preferred cutie")
        cutie_display_key = "consensus_masks_ds"

    focus_frames = []
    if "focus_frames" in options:
        focus_frames = options["focus_frames"]

    sleap_raw = sleap_data["raw"]["locations"]
    sleap_remaster = None
    if sleap_data['remaster'] != None:
        sleap_remaster = sleap_data["remaster"]["locations"]

    def pt_identity_xform(x):
        return int(x)
    def pt_downscale_xform(x):
        return int(x * scale_factor)
    def coord_identity_xform(xs):
        return np.round(xs).astype(int)
    def coord_downscale_xform(xs):
        return np.round(xs/4).astype(int)

    scale_factor = 1
    coord_xform = coord_identity_xform
    pt_xform = pt_identity_xform
    if downscaled:
        pt_xform = pt_downscale_xform
        coord_xform = coord_downscale_xform
        scale_factor = 0.25

    # We have sleap to cutie mappings (with sexes) in sleap_with_cutie_ident
    # { cutie trackid 0 = cutie data-id 1, cutie trackid 1 = cutie data-id 2 }
    # { cutie data-id 1 = Female, cutie data-id 2 = Male }
    c_sexmap = {
        0: "FEMALE",
        1: "MALE",
    }
    sex_colors = {
        "FEMALE": (120, 80, 225),
        "MALE": (210, 85, 25),
    }

    cap_video = cv2.VideoCapture(input_mp4)
    cap_video.set(cv2.CAP_PROP_POS_FRAMES, start)

    frame_width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_video.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_frame_width = int(scale_factor * frame_width)
    output_frame_height = int(scale_factor * frame_height)
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (output_frame_width, output_frame_height))

    chunkbase = start
    while chunkbase < start + length:
        thislength = min(start+length-chunkbase, CHUNK_LENGTH)
        cutie_data = cutie_masks[cutie_display_key][chunkbase:chunkbase+thislength]
        cutie_data = np.transpose(cutie_data, axes = [1,2,0])

        # separate cutie masks; {tracks, y, x, frames}
        masks = np.zeros((2, cutie_data.shape[0], cutie_data.shape[1], cutie_data.shape[2]))
        masks[0,:,:,:] = (cutie_data == 1).astype(np.uint8)
        masks[1,:,:,:] = (cutie_data == 2).astype(np.uint8)

        del cutie_data

        # iterate over length of mask data for mask and skeleton painting
        framenum = chunkbase
        chunkidx = 0
        while chunkidx < thislength:
            ret_video, frame_video = cap_video.read()
            if not ret_video:
                print(f"Can't read frame {framenum}")
                break

            if downscaled:
                frame_video = cv2.resize(frame_video,
                                         (int(frame_video.shape[1]*scale_factor),
                                             int(frame_video.shape[0]*scale_factor)),
                                         scale_factor, scale_factor,
                                         interpolation=cv2.INTER_NEAREST_EXACT)

            # Grab single frame's worth of mask info
            color_mask = np.zeros_like(frame_video)

            c0_sexcolor = sex_colors["FEMALE"]
            color_mask[masks[0, :, :, chunkidx] > 0, 0] = c0_sexcolor[0]
            color_mask[masks[0, :, :, chunkidx] > 0, 1] = c0_sexcolor[1]
            color_mask[masks[0, :, :, chunkidx] > 0, 2] = c0_sexcolor[2]

            c1_sexcolor = sex_colors["MALE"]
            color_mask[masks[1, :, :, chunkidx] > 0, 0] = c1_sexcolor[0]
            color_mask[masks[1, :, :, chunkidx] > 0, 1] = c1_sexcolor[1]
            color_mask[masks[1, :, :, chunkidx] > 0, 2] = c1_sexcolor[2]

            # Blend the video frame and the color mask
            alpha = 0.5  # Transparency factor
            overlaid_frame = cv2.addWeighted(frame_video, 1, color_mask, alpha, 0)

            # Plot SLEAP skeleton points
            def identity_xform(x):
                return int(x)
            def downscale_xform(x):
                return int(x * scale_factor)
            pt_xform = identity_xform
            if downscaled:
                pt_xform = downscale_xform

            annotation = []
            focus_frame = False
            for fcat, entries in focus_frames:
                if framenum in entries:
                    annotation.append(fcat)
                    focus_frame = True

            for slppt in range(sleap_with_cutie_ident.shape[1]):
                for anid in reversed(range(sleap_with_cutie_ident.shape[3])):
                    ptx, pty = sleap_with_cutie_ident[framenum, slppt, :, anid]
                    if ptx < 0:
                        continue
                    if np.isnan(ptx) or np.isnan(pty):
                        continue
                    ptx, pty = pt_xform(ptx), pt_xform(pty)

                    # default to a unmapped color
                    if anid == 2:
                        c_sexcolor = (75, 161, 0)
                        ptsize = 4
                    if anid == 3:
                        c_sexcolor = (152, 63, 127)
                        ptsize = 4

                    if anid in [0, 1]:
                        c_sexcolor = sex_colors[c_sexmap[anid]]
                        ptsize = 3
                    cv2.circle(overlaid_frame, (ptx, pty), ptsize+1, [255, 255, 255], -1)
                    cv2.circle(overlaid_frame, (ptx, pty), ptsize, c_sexcolor, -1)

            # Write the frame into the output video
            if font_backing:
                overlaid_frame = cv2.putText(overlaid_frame, f"Frame#: {framenum}",
                                     (25, 25), font, font_size, (0, 0, 0), font_thickness+1)
            overlaid_frame = cv2.putText(overlaid_frame, f"Frame#: {framenum}",
                                         (25, 25), font, font_size, font_main_color, font_thickness)

            if focus_frame:
                annotation = "+".join(annotation)
                if font_backing:
                    overlaid_frame = cv2.putText(overlaid_frame, f"Notes: {annotation}",
                                            (25, 50), font, font_size, (0, 0, 0), font_thickness+1)
                overlaid_frame = cv2.putText(overlaid_frame, f"Notes: {annotation}",
                                            (25, 50), font, font_size, font_main_color, font_thickness)
                for _ in range(50):
                    out.write(overlaid_frame)

            out.write(overlaid_frame)

            framenum += 1
            chunkidx += 1

        chunkbase += thislength

    # When everything is done, release the video capture and writer objects
    cap_video.release()
    out.release()

pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)

vidname = exp_with_ext

sleap_data = {}
sleap_data["raw"] = getSleapData(args.sleap_path)
if args.sleap_remaster_path != "":
    sleap_data["remaster"] = getSleapData(args.sleap_remaster_path)
else:
    sleap_data["remaster"] = None

masks_fwd_path = os.path.abspath(os.path.join(args.cutie_path, "masks_forward.h5"))
masks_rev_path = os.path.abspath(os.path.join(args.cutie_path, "masks_reverse.h5"))
masks_fwd_fh = h5py.File(masks_fwd_path, "r")
masks_rev_fh = h5py.File(masks_rev_path, "r")
cutie_masks = {
    "masks_fwd_ds": masks_fwd_fh["masks"],
    "masks_rev_ds": masks_rev_fh["masks"],
}

frame_count = min(sleap_data["raw"]["locations"].shape[0], masks_fwd_fh["masks"].shape[0])

zod_df, cutie_animal_mappings = resolveAnnotations(frame_count, args.annotations)

# export consensus cutie if not already
consensus_masks_path = os.path.abspath(os.path.join(args.output_path, f"{exp}.cutie.consensus.h5"))
rejected_masks_path = os.path.abspath(os.path.join(args.output_path, f"{exp}.cutie.rejected.h5"))
export_consensus_cutie(
    consensus_h5_path=consensus_masks_path,
    rejected_h5_path=rejected_masks_path,
    start=0,
    length=frame_count,
    cutie_masks=cutie_masks,
    downscaled=True)

consensus_masks_fh = h5py.File(consensus_masks_path, "r")
cutie_masks["consensus_masks_ds"] = consensus_masks_fh["masks"]
rejected_masks_fh = h5py.File(rejected_masks_path, "r")
cutie_masks["rejected_masks_ds"] = rejected_masks_fh["masks"]

sleap_to_consensus_cutie_masks_path = os.path.join(args.output_path, f"{exp}.sleap_x_consensus_cutie_mapping.npy")
if os.path.exists(sleap_to_consensus_cutie_masks_path):
    print("Loading sleap<=>cutie mask tables.")
    sleap_to_consensus_cutie_masks = np.load(sleap_to_consensus_cutie_masks_path)
    print(sleap_to_consensus_cutie_masks.shape)
else:
    sleap_to_consensus_cutie_masks = sleap_to_consensus_cutie_mapping(
                                        sleap_data=sleap_data,
                                        cutie_masks=cutie_masks,
                                        dilation_kernel=5, downscaled=True)
    print("Saving sleap<=>consensus_cutie mask tables.")
    print(sleap_to_consensus_cutie_masks.shape)
    np.save(sleap_to_consensus_cutie_masks_path, sleap_to_consensus_cutie_masks)

print(f"Sleap<=>Cutie map shape: {sleap_to_consensus_cutie_masks.shape}")

sleap_save_path = os.path.join(args.output_path, f"{exp}.aligned.preview.infill.npy")
infill_metadata = align_sleap(sleap_data=sleap_data,
                              sleap_save_path=sleap_save_path,
                              cutie_masks=cutie_masks,
                              cutie_animal_mappings=cutie_animal_mappings,
                              sleap_to_consensus_cutie_masks=sleap_to_consensus_cutie_masks)
sleap_with_cutie_ident = infill_metadata["sleap_with_cutie_ident"]

# we only need to do more work if we want videos/supercuts
if not do_video:
    sys.exit(0)

chunks_to_process = []
chunks_to_process.append({
        "start": 0,
        "length": frame_count,
        "output_path": os.path.join(args.output_path, f"{exp}.overlayed.mp4"),
        "options": { "options": ["standard_cutie_track"] },
    })

drawn_frame_tracking = np.zeros(frame_count)

# for each zod_df item, let's get it and +/- 100 frames.
zod_df = zod_df.sort_values("region_start")
for zod_idx in range(len(zod_df)):
    r_start = zod_df.loc[zod_idx, 'region_start']
    r_end = zod_df.loc[zod_idx, 'region_end']

    drawn_frame_tracking[r_start:r_end+1] = 1

    rr_start = max(0, r_start - 100)
    rr_end = min(frame_count - 1, r_end + 100)

    continue
    chunks_to_process.append({
            "start": rr_start,
            "length": rr_end - rr_start + 1,
            "output_path": os.path.join(args.output_path, f"{exp}.section_{r_start}x{r_end}.mp4"),
            "options": { "options": ["standard_cutie_track"] },
        })
    chunks_to_process.append({
            "start": rr_start,
            "length": rr_end - rr_start + 1,
            "output_path": os.path.join(args.output_path, f"{exp}.section_{r_start}x{r_end}.rejected_cutie_track.mp4"),
            "options": { "options": ["rejected_cutie_track"] },
        })

# let's also display the sections of infill that weren't deliberately drawn as part of a ZOB
remainder_categories = [["absent_cutie", infill_metadata["remainder_absent_cutie"]],
                        ["absent_sleap", infill_metadata["remainder_absent_sleap"]],
                        ["non_overlap", infill_metadata["remainder_non_overlap"]],
                        ["ambiguous", infill_metadata["remainder_nontrivial"]]]
for cat, targets in remainder_categories:
    for target in targets:

        #skip if already drawn
        if drawn_frame_tracking[target] > 0:
            continue

        rr_start = max(0, target - 100)
        rr_end = min(frame_count - 1, target + 100)

        chunks_to_process.append({
            "start": rr_start,
            "length": rr_end - rr_start + 1,
            "output_path": os.path.join(args.output_path, f"{exp}.err_infills_{rr_start}x{rr_end}.mp4"),
            "options": { "options": ["standard_cutie_track"], "focus_frames": remainder_categories },
        })

        drawn_frame_tracking[rr_start:rr_end+1] = 1

for cidx in range(len(chunks_to_process)):
    s = chunks_to_process[cidx]["start"]
    e = chunks_to_process[cidx]["start"] + chunks_to_process[cidx]["length"]
    idx = np.arange(s, e)

    plt.figure()
    plt.plot(idx, cutie_animal_mappings["cutie_fwd_valid"][s:e], label="fwd_v")
    plt.plot(idx, cutie_animal_mappings["cutie_rev_valid"][s:e]+.05, label="rev_v")
    plt.plot(idx, cutie_animal_mappings["cutie_preferred"][s:e]+.10, label="pref")
    plt.legend()
    plt.savefig(os.path.join(chunks_to_process[cidx]["output_path"]+".mappings.png"))
    plt.clf()
    plt.close()

    draw_section(input_mp4=args.video_path,
                 output_mp4=chunks_to_process[cidx]["output_path"],
                 start=chunks_to_process[cidx]["start"],
                 length=chunks_to_process[cidx]["length"],
                 sleap_data=sleap_data,
                 cutie_masks=cutie_masks,
                 sleap_with_cutie_ident=sleap_with_cutie_ident,
                 cutie_animal_mappings=cutie_animal_mappings,
                 sleap_to_consensus_cutie_masks=sleap_to_consensus_cutie_masks,
                 options=chunks_to_process[cidx]["options"],
                 downscaled=True)
    print(f"Completed cidx {cidx} of {len(chunks_to_process)}")

print("DONE!")
