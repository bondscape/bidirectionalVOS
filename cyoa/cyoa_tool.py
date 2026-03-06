import shutil
import cv2
import numpy as np
import os
import sys
import json
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import simpledialog
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import find_objects, label as label_objects

import argparse

INPUT_DELAY_MS = 5

#anColorsSaved = [[60, 60, 0], [0, 60, 60], [0, 0, 120]]
anColors = [[128, 128, 0], [0, 128, 128], [180, 20, 180], [120, 0, 0], [0, 120, 0], [0, 0, 120]]
uiColors = [[128, 128, 0], [0, 128, 128], [180, 20, 180], [0, 180, 60]]

#simplified
font = cv2.FONT_HERSHEY_TRIPLEX
font_size = 0.45
font_thickness = 1
font_backing = False
font_main_color = (140, 30, 70)
window_scale = 0.5

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", help="path to video.mp4", required=True)
parser.add_argument("--cutie_path", help="path to cutie segments folder", required=True)
parser.add_argument("--sleap_path", help="path to sleap tracked.cleaned.h5 file", required=True)

parser.add_argument("--region_size", default="500")
parser.add_argument("--quality_threshold", default="0.3")
parser.add_argument("--show_alt", default="0")
parser.add_argument("--rescan", action="store_true")

args = parser.parse_args()

exp = os.path.splitext(os.path.basename(args.video_path))[0]
exp_with_ext = os.path.basename(args.video_path)

dataframe_for_results = f"{exp}.annotations.csv"
dataframe_for_results_rescan = f"{exp}.annotations.rescan.csv"

rescan_mode = False
if args.rescan:
    rescan_mode = True

region_size = int(args.region_size)
QUALITY_THRESHOLD = float(args.quality_threshold)
show_alt = int(args.show_alt) > 0

print(f"Exp: {exp}")

cv2.namedWindow("viewer", cv2.WINDOW_NORMAL)
if show_alt:
    cv2.namedWindow("alt_viewer", cv2.WINDOW_NORMAL)

def saveAll(colorframe):
    import time
    time.sleep(1)
    fname = f"screenshot-{int(time.time())}.png"
    cv2.imwrite(fname, colorframe)

def drawSLEAPPositions(colorframe, ds_factor, sleap_locations, marker_size):
    for aid in range(sleap_locations.shape[2]):
        for ptid in range(sleap_locations.shape[0]):
            if np.isnan(sleap_locations[ptid, 0, aid]):
                continue
            loc_x, loc_y = (sleap_locations[ptid, :, aid] / ds_factor).astype(int)
            if loc_x < 0 or loc_y < 0:
                continue
            colorframe = cv2.circle(colorframe, (loc_x, loc_y), marker_size, anColors[aid], -1)
    return colorframe

def drawCutieMasks(colorframe, downscaled_masks, frame_num, section_start, section_end):
    overlayed = np.zeros((colorframe.shape[0], colorframe.shape[1], 3), dtype=np.uint8)
    overlayed[:colorframe.shape[0], :colorframe.shape[1]] = colorframe

    mask_upscale_factor = float(colorframe.shape[0] / downscaled_masks[frame_num].shape[0])
    masks = downscaled_masks[frame_num]

    masks = cv2.resize(masks, (colorframe.shape[1], colorframe.shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)

    for id in [1, 2]:
        maskForId = np.zeros((overlayed.shape[0], overlayed.shape[1]), dtype=np.uint8)
        maskForId[:masks.shape[0],:] = (masks == id).astype(np.uint8)
        palette = np.zeros((overlayed.shape[0], overlayed.shape[1], 3), dtype=np.uint8)
        palette[:,:] = np.array(anColors[id]) #fixme
        palette = cv2.bitwise_and(palette, palette, mask=maskForId)
        overlayed = cv2.add(overlayed, (palette*.7).astype(np.uint8))

    if abs(section_start - frame_num) < 15 or abs(section_end - frame_num) < 15:
        c1i = np.argwhere(downscaled_masks[frame_num] == 1)
        if len(c1i) > 0:
            c1 = np.mean(c1i, axis=0).astype(int)
            overlayed = cv2.circle(overlayed,
                                   (int(mask_upscale_factor * c1[1]),
                                    int(mask_upscale_factor * c1[0])),
                                   7, anColors[3], -1)
        c2i = np.argwhere(downscaled_masks[frame_num] == 2)
        if len(c2i) > 0:
            c2 = np.mean(c2i, axis=0).astype(int)
            overlayed = cv2.circle(overlayed,
                                   (int(mask_upscale_factor * c2[1]),
                                    int(mask_upscale_factor * c2[0])),
                                   7, anColors[4], -1)

    return overlayed

def drawProgressOverlay(frame, frame_num, start, end):
    percent = ((frame_num - start) / (end - start + 1)) * 100.0
    if font_backing:
        frame = cv2.putText(frame,
                            f"Progress: {percent:02.0f}% [{start}:{frame_num}:{end}]",
                            (25, 25), font,
                            font_size, (0, 0, 0), font_thickness+1, cv2.LINE_AA)
    frame = cv2.putText(frame,
                        f"Progress: {percent:02.0f}% [{start}:{frame_num}:{end}]",
                        (25, 25), font,
                        font_size, font_main_color, font_thickness, cv2.LINE_AA)
    if frame_num >= start and frame_num <= end:
        if font_backing:
            frame = cv2.putText(frame,
                                f"* ZOD *",
                                (25, 100), font,
                                font_size, (0, 0, 0), font_thickness+1, cv2.LINE_AA)
        frame = cv2.putText(frame,
                            "* ZOD *",
                            (25, 100), font,
                            font_size, (0, 0, 120), font_thickness, cv2.LINE_AA)
    return frame

def drawHeader(frame, question):
    max_y = frame.shape[0]
    if font_backing:
        frame = cv2.putText(frame,
                            question,
                            (25, 50), font,
                            font_size, (0, 0, 0), font_thickness+1, cv2.LINE_AA)
    frame = cv2.putText(frame,
                        question,
                        (25, 50), font,
                        font_size, font_main_color, font_thickness, cv2.LINE_AA)
    return frame

def drawQuestion(frame, line_number, question):
    max_y = frame.shape[0]
    if font_backing:
        frame = cv2.putText(frame,
                            question,
                            (25, max_y - 65 + (28 * line_number)), font,
                            font_size, (0, 0, 0), font_thickness+1, cv2.LINE_AA)
    frame = cv2.putText(frame,
                        question,
                        (25, max_y - 65 + (28 * line_number)), font,
                        font_size, font_main_color, font_thickness, cv2.LINE_AA)
    return frame


def swapStr(isSwapped):
    if isSwapped:
        return "SWAPPED"
    return "STABLE"

def cycleWindowMode(windowMode):
    if windowMode == "ZOD":
        return "PREVIEW"
    if windowMode == "PREVIEW":
        return "ZOD"
    print(f"Unknown windowSelection mode: {windowMode}")
    sys.exit(1)

def cycleGarbage(isGarbage):
    if isGarbage == "GOOD":
        isGarbage = "GAUCHE"
    elif isGarbage == "GAUCHE":
        isGarbage = "GARBAGE"
    elif isGarbage == "GARBAGE":
        isGarbage = "GOOD"
    else:
        print(f"Unrecognized value for isGarbage: {isGarbage}")
        sys.exit(1)
    return isGarbage

def askCutieQuestions(thiszod, mp4_in, exp, direction, masks_dir_ds, masks_altdir_ds):
    global INPUT_DELAY_MS
    global fig_local
    global section_start
    global section_end
    global preview_start
    global preview_end
    global control_dirtyplot
    global control_seeked
    global control_figure_section_selector
    global control_window_mode

    control_figure_section_selector = "LEFT"

    playing = True
    lzod = section_end - section_start + 1
    isSwapped = False
    isGarbage = "GOOD"

    DOWNSCALE_FACTOR = 4
    vh = cv2.VideoCapture(mp4_in)

    cv2.resizeWindow("viewer",
                    int(vh.get(cv2.CAP_PROP_FRAME_WIDTH) * window_scale),
                    int(vh.get(cv2.CAP_PROP_FRAME_HEIGHT) * window_scale))
    if show_alt:
        cv2.resizeWindow("alt_viewer",
                        int(vh.get(cv2.CAP_PROP_FRAME_WIDTH) * window_scale),
                        int(vh.get(cv2.CAP_PROP_FRAME_HEIGHT) * window_scale))

    unanswered = True
    control_seeked = True
    toggleText = False
    frame_num = preview_start
    saveFrame = False
    while unanswered:
            if control_dirtyplot:
                control_dirtyplot = False
                redraw_local_plot(thiszod, section_start, section_end, preview_start, preview_end)

            if control_seeked:
                vh.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ok, vidframe = vh.read()
                if not ok:
                    print(f"***** ERROR reading frame {frame_num}")
                    break
                vh.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                control_seeked = False

            if playing:
                ok, vidframe = vh.read()
                if not ok:
                    print(f"***** ERROR reading frame {frame_num}")
                    break
            frame = cv2.resize(vidframe,
                               (int(vidframe.shape[1] / DOWNSCALE_FACTOR),
                                int(vidframe.shape[0] / DOWNSCALE_FACTOR)),
                               0, 0, interpolation=cv2.INTER_NEAREST_EXACT)
            if show_alt:
                altframe = frame.copy()
            frame_with_overlay = drawCutieMasks(frame, masks_dir_ds,
                                                frame_num, section_start, section_end)


            if not toggleText:
                frame_with_overlay = drawProgressOverlay(frame_with_overlay, frame_num,
                                                         section_start, section_end)

                frame_with_overlay = drawQuestion(frame_with_overlay, 2,
                                                f"Mask identities [S]wapped/[S]table? {swapStr(isSwapped)}")
                frame_with_overlay = drawQuestion(frame_with_overlay, 1,
                                                f"Mask quality [G]ood/[G]auche/[G]arbage? {isGarbage}")
                frame_with_overlay = drawQuestion(frame_with_overlay, 0,
                                                f"[Esc] - cancel, [Enter] - continue")
                frame_with_overlay = drawHeader(frame_with_overlay, f"{exp} - {direction} cutie. l(ZOD)={lzod}. WQ={1-locquality}")
            cv2.imshow("viewer", frame_with_overlay)

            if show_alt:
                altframe_with_overlay = drawCutieMasks(altframe, masks_altdir_ds,
                                                       frame_num, section_start, section_end)
                alt_frame_with_overlay = drawHeader(altframe_with_overlay,
                                                    f"==== ALTDIR cutie masks ====")
                cv2.imshow("alt_viewer", alt_frame_with_overlay)

            if saveFrame:
                if show_alt:
                    saveAll(alt_frame_with_overlay)
                else:
                    saveAll(frame_with_overlay)
                saveFrame = False
            keyres = cv2.waitKey(INPUT_DELAY_MS)
            if (keyres & 0xFF) == ord('-'):
                INPUT_DELAY_MS += 1
            if (keyres & 0xFF) in [ord('='), ord('+')]:
                INPUT_DELAY_MS = max(INPUT_DELAY_MS - 1, 1)
            if (keyres & 0xFF) == ord("'"):
                side = control_figure_section_selector
                strbound = simpledialog.askstring(f"Please enter {control_window_mode} {side} boundary",
                                                  control_window_mode)
                try:
                    bound = int(strbound)
                    update_boundaries(bound)
                except:
                    print(f"Rejected input: {strbound}")
            if (keyres & 0xFF) == ord("t"):
                toggleText = not toggleText
            if (keyres & 0xFF) == ord("c"):
                zoomOut()
            if (keyres & 0xFF) == ord("v"):
                zoomIn()
            if (keyres & 0xFF) == ord("["):
                control_seeked = True
                frame_num = max(0, frame_num - 1)
                continue
            if (keyres & 0xFF) == ord("]"):
                control_seeked = True
                frame_num = min(video_length, frame_num + 1)
                continue
            if (keyres & 0xFF) == ord(","):
                control_seeked = True
                frame_num = section_start
                continue
            if (keyres & 0xFF) == ord("."):
                control_seeked = True
                frame_num = section_end
                continue
            if (keyres & 0xFF) == ord("s"):
                isSwapped = not isSwapped
            if (keyres & 0xFF) == ord(" "):
                playing = not playing
            if (keyres & 0xFF) == ord("g"):
                isGarbage = cycleGarbage(isGarbage)
            if (keyres & 0xFF) == ord("z"):
                control_window_mode = cycleWindowMode(control_window_mode)
                control_dirtyplot = True
            if (keyres & 0xFF) == ord('/'):
                saveFrame = True
            if (keyres & 0xFF) == 27:
                sys.exit(0)
            if (keyres & 0xFF) in [8, 127]:
                return "BACK", 0, 0
            if (keyres & 0xFF) == ord('q'):
                return "SKIP", 0, 0
            if (keyres & 0xFF) == 13:
                return "CONTINUE", swapStr(isSwapped), isGarbage
            if playing:
                frame_num += 1
            if frame_num > preview_end:
                frame_num = preview_start
                control_seeked = True

def askSleapQuestions(thiszod, mp4_in, exp, sleap_locations):
    global INPUT_DELAY_MS
    global section_start
    global section_end
    global preview_start
    global preview_end
    global control_dirtyplot
    global control_seeked
    global control_figure_section_selector
    global control_window_mode

    control_figure_section_selector = "LEFT"

    playing = True
    lzod = section_end - section_start + 1
    isSwapped = False
    isGarbage = "GOOD"
    vh = cv2.VideoCapture(mp4_in)
    cv2.resizeWindow("viewer",
                     int(vh.get(cv2.CAP_PROP_FRAME_WIDTH) * window_scale),
                     int(vh.get(cv2.CAP_PROP_FRAME_HEIGHT) * window_scale))
    DOWNSCALE_FACTOR = 4
    unanswered = True
    control_seeked = True
    frame_num = preview_start
    saveFrame = False
    while unanswered:
            if control_dirtyplot:
                control_dirtyplot = False
                redraw_local_plot(thiszod, section_start, section_end, preview_start, preview_end)

            if control_seeked:
                vh.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ok, vidframe = vh.read()
                if not ok:
                    print(f"***** ERROR reading frame {frame_num}")
                    break
                vh.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                control_seeked = False

            if playing:
                ok, vidframe = vh.read()
                if not ok:
                    print(f"***** ERROR reading frame {frame_num}")
                    break
            frame_with_overlay = cv2.resize(vidframe,
                                            (int(vidframe.shape[1] / DOWNSCALE_FACTOR),
                                                int(vidframe.shape[0] / DOWNSCALE_FACTOR)),
                                            0, 0, interpolation=cv2.INTER_NEAREST_EXACT)

            slp_size = 3
            if frame_num >= section_start and frame_num <= section_end:
                slp_size = 5
            frame_with_overlay = drawSLEAPPositions(frame_with_overlay, DOWNSCALE_FACTOR,
                                                    sleap_locations[frame_num], slp_size)
            frame_with_overlay = drawProgressOverlay(frame_with_overlay, frame_num, section_start, section_end)
            frame_with_overlay = drawQuestion(frame_with_overlay, 0,
                                              f"Sleap identities [S]wapped/[S]table? {swapStr(isSwapped)}")
            frame_with_overlay = drawQuestion(frame_with_overlay, 1,
                                              f"SLEAP quality [G]ood/[G]auche/[G]arbage? {isGarbage}")
            frame_with_overlay = drawQuestion(frame_with_overlay, 2,
                                              f"[Esc] - cancel, [Enter] - continue")
            frame_with_overlay = drawHeader(frame_with_overlay, f"{exp} - sleap quality. l(ZOD)={lzod}")
            cv2.imshow("viewer", frame_with_overlay)

            if saveFrame:
                saveAll(frame_with_overlay)
                saveFrame = False
            keyres = cv2.waitKey(INPUT_DELAY_MS)
            if (keyres & 0xFF) == ord('-'):
                INPUT_DELAY_MS += 1
            if (keyres & 0xFF) in [ord('='), ord('+')]:
                INPUT_DELAY_MS = max(INPUT_DELAY_MS - 1, 1)
            if (keyres & 0xFF) == ord("'"):
                side = control_figure_section_selector
                strbound = simpledialog.askstring(f"Please enter {control_window_mode} {side} boundary",
                                                  control_window_mode)
                try:
                    bound = int(strbound)
                    update_boundaries(bound)
                except:
                    print(f"Rejected input: {strbound}")
            if (keyres & 0xFF) == ord("c"):
                zoomOut()
            if (keyres & 0xFF) == ord("v"):
                zoomIn()
            if (keyres & 0xFF) == ord("["):
                control_seeked = True
                frame_num = max(0, frame_num - 1)
                continue
            if (keyres & 0xFF) == ord("]"):
                control_seeked = True
                frame_num = min(video_length, frame_num + 1)
                continue
            if (keyres & 0xFF) == ord(","):
                control_seeked = True
                frame_num = section_start
                continue
            if (keyres & 0xFF) == ord("."):
                control_seeked = True
                frame_num = section_end
                continue
            if (keyres & 0xFF) == ord("s"):
                isSwapped = not isSwapped
            if (keyres & 0xFF) == ord(" "):
                playing = not playing
            if (keyres & 0xFF) == ord("g"):
                isGarbage = cycleGarbage(isGarbage)
            if (keyres & 0xFF) == ord("z"):
                control_window_mode = cycleWindowMode(control_window_mode)
                control_dirtyplot = True
            if (keyres & 0xFF) == ord('/'):
                saveFrame = True
            if (keyres & 0xFF) == 27:
                sys.exit(0)
            if (keyres & 0xFF) in [8, 127]:
                return "BACK", 0, 0
            if (keyres & 0xFF) == ord('q'):
                return "SKIP", 0, 0
            if (keyres & 0xFF) == 13:
                return "CONTINUE", swapStr(isSwapped), isGarbage
            if playing:
                frame_num += 1
            if frame_num > preview_end:
                frame_num = preview_start
                control_seeked = True

def interpolate_missing(Y, kind="linear", subset=None):
    """Fills missing values independently along each dimension after the first."""

    if subset is None:

        # Store initial shape.
        initial_shape = Y.shape
        print(f"Initial shape: {initial_shape}")

        # Flatten after first dim.
        Y = Y.reshape((initial_shape[0], -1))

        # Interpolate along each slice.
        print(f"Y shape:")
        print(Y.shape)
        for i in range(Y.shape[-1]):
            y = Y[:, i]

            # Build interpolant.
            x = np.flatnonzero(~np.isnan(y))
            f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

            # Fill missing
            xq = np.flatnonzero(np.isnan(y))
            y[xq] = f(xq)

            # Fill leading or trailing NaNs with the nearest non-NaN values
            mask = np.isnan(y)
            y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

            # Save slice
            Y[:, i] = y

        # Restore to initial shape.
        Y = Y.reshape(initial_shape)
    return Y


def askInterpolatedSleapQuestions(thiszod, mp4_in, exp, sleap_locations, swappedStatus):
    global INPUT_DELAY_MS
    global section_start
    global section_end
    global preview_start
    global preview_end
    global control_dirtyplot
    global control_seeked
    global control_figure_section_selector
    global control_window_mode

    control_figure_section_selector = "LEFT"

    playing = True
    lzod = section_end - section_start + 1
    # zero out the bad stuff
    sleap_locations = sleap_locations.copy()
    sleap_locations[section_start:section_end] = np.nan
    if swappedStatus == "SWAPPED":
        sleap_locations[section_end:,:,:,:] = sleap_locations[section_end:,:,:,::-1]
    sleap_locations = interpolate_missing(sleap_locations)

    isGarbage = "GOOD"
    DOWNSCALE_FACTOR = 4
    vh = cv2.VideoCapture(mp4_in)
    cv2.resizeWindow("viewer",
                     int(vh.get(cv2.CAP_PROP_FRAME_WIDTH) * window_scale),
                     int(vh.get(cv2.CAP_PROP_FRAME_HEIGHT) * window_scale))
    unanswered = True
    control_seeked = True
    frame_num = preview_start
    saveFrame = False
    while unanswered:
            if control_dirtyplot:
                control_dirtyplot = False
                redraw_local_plot(thiszod, section_start, section_end, preview_start, preview_end)

            if control_seeked:
                vh.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ok, vidframe = vh.read()
                if not ok:
                    print(f"***** ERROR reading frame {frame_num}")
                    break
                vh.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                control_seeked = False

            if playing:
                ok, vidframe = vh.read()
                if not ok:
                    print(f"***** ERROR reading frame {frame_num}")
                    break
            frame_with_overlay = cv2.resize(vidframe,
                                            (int(vidframe.shape[1] / DOWNSCALE_FACTOR),
                                                int(vidframe.shape[0] / DOWNSCALE_FACTOR)),
                                            0, 0, interpolation=cv2.INTER_NEAREST_EXACT)

            slp_size = 3
            if frame_num >= section_start and frame_num <= section_end:
                slp_size = 5
            frame_with_overlay = drawSLEAPPositions(frame_with_overlay, DOWNSCALE_FACTOR,
                                                    sleap_locations[frame_num], slp_size)
            frame_with_overlay = drawProgressOverlay(frame_with_overlay, frame_num, section_start, section_end)
            frame_with_overlay = drawQuestion(frame_with_overlay, 1,
                                              f"SLEAP quality [G]ood/[G]auche/[G]arbage? {isGarbage}")
            frame_with_overlay = drawQuestion(frame_with_overlay, 2,
                                              f"[Esc] - cancel, [Enter] - continue")
            frame_with_overlay = drawHeader(frame_with_overlay, f"{exp} - sleap interpolation. l(ZOD)={lzod}")
            cv2.imshow("viewer", frame_with_overlay)

            if saveFrame:
                saveAll(frame_with_overlay)
                saveFrame = False
            keyres = cv2.waitKey(INPUT_DELAY_MS)
            if (keyres & 0xFF) == ord('-'):
                INPUT_DELAY_MS += 1
            if (keyres & 0xFF) in [ord('='), ord('+')]:
                INPUT_DELAY_MS = max(INPUT_DELAY_MS - 1, 1)
            if (keyres & 0xFF) == ord("'"):
                side = control_figure_section_selector
                strbound = simpledialog.askstring(f"Please enter {control_window_mode} {side} boundary",
                                                  control_window_mode)
                try:
                    bound = int(strbound)
                    update_boundaries(bound)
                except:
                    print(f"Rejected input: {strbound}")
            if (keyres & 0xFF) == ord("c"):
                zoomOut()
            if (keyres & 0xFF) == ord("v"):
                zoomIn()
            if (keyres & 0xFF) == ord("["):
                control_seeked = True
                frame_num = max(0, frame_num - 1)
                continue
            if (keyres & 0xFF) == ord("]"):
                control_seeked = True
                frame_num = min(video_length, frame_num + 1)
                continue
            if (keyres & 0xFF) == ord(","):
                control_seeked = True
                frame_num = section_start
                continue
            if (keyres & 0xFF) == ord("."):
                control_seeked = True
                frame_num = section_end
                continue
            if (keyres & 0xFF) == ord(" "):
                playing = not playing
            if (keyres & 0xFF) == ord("g"):
                isGarbage = cycleGarbage(isGarbage)
            if (keyres & 0xFF) == ord("z"):
                control_window_mode = cycleWindowMode(control_window_mode)
                control_dirtyplot = True
            if (keyres & 0xFF) == ord('/'):
                saveFrame = True
            if (keyres & 0xFF) == 27:
                sys.exit(0)
            if (keyres & 0xFF) in [8, 127]:
                return "BACK", 0
            if (keyres & 0xFF) == ord('q'):
                return "SKIP", 0
            if (keyres & 0xFF) == 13:
                return "CONTINUE", isGarbage
            if playing:
                frame_num += 1
            if frame_num > preview_end:
                frame_num = preview_start
                control_seeked = True

def navigateSection(exp, zod, mp4_in, masks_fwd_path, masks_rev_path, sleap_path):
    global section_start
    global section_end
    global preview_start
    global preview_end

    masks_fwd_fh = h5py.File(masks_fwd_path, "r")
    masks_rev_fh = h5py.File(masks_rev_path, "r")
    masks_fwd_ds = masks_fwd_fh["masks"]
    masks_rev_ds = masks_rev_fh["masks"]
    if masks_fwd_ds.shape[0] != masks_rev_ds.shape[0]:
        print(f"****** WARNING! Video {mp4_in} has unequal mask ranges {masks_rev_ds.shape} vs {masks_fwd_ds.shape}!")

    best_cutie = ""
    best_cutie_swapped = ""
    best_cutie_quality = ""
    slp_raw_qual = ""
    slp_raw_swapped = ""
    interpolated_sleap = False

    states = ["CUTIE_F", "CUTIE_R", "SLEAP", "SLEAP_INT", "END"]
    sidx = 0
    while True:
        state = states[sidx]
        if state == "END":
            print("Saving results...")
            break

        if state == "CUTIE_F":
            # visual check 1: did the cutie masks flip during the clip?
            cmd, c_fwd_swapped, c_fwd_qual = askCutieQuestions(zod, mp4_in, exp, "forward", masks_fwd_ds, masks_rev_ds)
            if cmd == "SKIP":
                return False, {}
            if cmd == "BACK":
                # there is no reverse from here, just try again.
                continue
            if cmd == "CONTINUE":
                sidx += 1
                continue
        if state == "CUTIE_R":
            cmd, c_rev_swapped, c_rev_qual = askCutieQuestions(zod, mp4_in, exp, "reverse", masks_rev_ds, masks_fwd_ds)
            if cmd == "SKIP":
                return False, {}
            if cmd == "BACK":
                sidx -= 1
                continue
            if cmd == "CONTINUE":
                sidx += 1
                continue
        if state == "SLEAP":
            if c_fwd_qual == "GARBAGE" and c_rev_qual == "GARBAGE":
                print("No good cutie for section.")
                best_cutie = "NONE"
                best_cutie_quality = "GARBAGE"
                best_cutie_swapped = ""
            else:
                if (c_fwd_qual == "GOOD" and c_rev_qual != "GOOD") or \
                (c_fwd_qual == "GAUCHE" and not c_rev_qual in ["GOOD", "GAUCHE"]):
                    best_cutie = "FORWARD"
                    best_cutie_quality = c_fwd_qual
                    best_cutie_swapped = c_fwd_swapped
                elif c_rev_qual in ["GAUCHE", "GOOD"]:
                    best_cutie = "REVERSE"
                    best_cutie_quality = c_rev_qual
                    best_cutie_swapped = c_rev_swapped
                elif c_fwd_qual == "GARBAGE":
                    best_cutie = "REVERSE"
                    best_cutie_quality = c_rev_qual
                    best_cutie_swapped = c_rev_swapped

            print(f"Cutie: {best_cutie}. Cutie-swap: {best_cutie_swapped}")
            # sleap data
            with h5py.File(sleap_path) as fh:
                raw_sleap_locations = fh["tracks"][:].T
                raw_node_names = list(map(lambda x: x.decode(), fh["node_names"][:].T))

            # superimpose sleap mapping. was there a swap?
            cmd, slp_raw_swapped, slp_raw_qual = askSleapQuestions(zod, mp4_in, exp, raw_sleap_locations)
            if cmd == "SKIP":
                return False, {}
            if cmd == "BACK":
                sidx -= 1
                continue
            if cmd == "CONTINUE":
                sidx += 1
                continue
        if state == "SLEAP_INT":
            interpolated_sleap = False
            if slp_raw_qual == "GARBAGE" or slp_raw_qual == "GAUCHE":
                interpolated_sleap = True
                cmd, slp_int_qual = askInterpolatedSleapQuestions(zod, mp4_in, exp, raw_sleap_locations,
                                                                    slp_raw_swapped)
                if cmd == "SKIP":
                    return False, {}
                if cmd == "BACK":
                    sidx -= 1
                    continue
                if cmd == "CONTINUE":
                    sidx += 1
                    continue
            else:
                sidx += 1
                continue

    results = {
                  "cutie_forward_quality": c_fwd_qual,
                  "cutie_forward_swapped": c_fwd_swapped,
                  "cutie_reverse_quality": c_rev_qual,
                  "cutie_reverse_swapped": c_rev_swapped,
                  "best_cutie": best_cutie,
                  "best_cutie_swapped": best_cutie_swapped,
                  "best_cutie_quality": best_cutie_quality,
                  "sleap_raw_quality": slp_raw_qual,
                  "sleap_raw_swapped": slp_raw_swapped,
              }
    if interpolated_sleap:
        results["sleap_interpolated_quality"] = slp_int_qual
    return True, results

def load_zod_csv(exp):
    df = pd.DataFrame(columns = ["region_start", "region_end", "source", "previously_scored", "cutie_forward_quality", "cutie_forward_swapped", "cutie_reverse_quality", "cutie_reverse_swapped", "best_cutie", "best_cutie_swapped", "best_cutie_quality", "sleap_raw_quality", "sleap_raw_swapped", "sleap_interpolated_quality", "proofed"])

    has_existing = False
    has_rescan = False
    if os.path.exists(dataframe_for_results):
        print("Existing annotations.csv")
        has_existing = True
    if os.path.exists(dataframe_for_results_rescan):
        print("Existing annotations.rescan.csv")
        has_rescan = True

    if rescan_mode:
        if not has_existing:
            print(f"Expected original {dataframe_for_results} to exist!")
            sys.exit(1)
        if has_rescan:
            df = pd.read_csv(dataframe_for_results_rescan, keep_default_na=False)
        else:
            df = pd.read_csv(dataframe_for_results, keep_default_na=False)
        print(df)
        if "previously_scored" not in df.columns:
            df["previously_scored"] = np.ones(len(df))
        if "source" not in df.columns:
            df["source"] = ["UNKNOWN"] * len(df)
        if "proofed" not in df.columns:
            df["proofed"] = [""] * len(df)
    else:
        if has_existing:
            df = pd.read_csv(dataframe_for_results, keep_default_na=False)
        else:
            print("Starting from empty ZOD inventory")

    # clean up some older errors
    for idx in range(len(df)):
        if "SWAPS" in df.loc[idx, "source"]:
            df.loc[idx, "source"] = "SWAPS"
        if "WORSTLOCS" in df.loc[idx, "source"]:
            df.loc[idx, "source"] = "WORSTLOCS"

    df["region_start"] = df["region_start"].astype(int)
    df["region_end"] = df["region_end"].astype(int)
    df["previously_scored"] = df["previously_scored"].astype(int)

    return df


mp4_in = args.video_path
masks_fwd_path = os.path.abspath(os.path.join(args.cutie_path, "masks_forward.h5"))
masks_rev_path = os.path.abspath(os.path.join(args.cutie_path, "masks_reverse.h5"))
sleap_path = os.path.abspath(args.sleap_path)

iou = np.load(os.path.join(args.cutie_path, "iou.npy"))
alt_iou = np.load(os.path.join(args.cutie_path, "alt_iou.npy"))
track_quality = (1.0 - np.abs(iou-alt_iou))
candidate_search_quality = track_quality.copy()

video_length = iou.shape[0]

previously_scored_regions = np.zeros_like(track_quality)
if rescan_mode:
    df = load_zod_csv(exp)
    # create a filter for anywhere that's been previously scored.
    for idx in range(len(df.index)):
        row = df.loc[idx]
        if int(row["previously_scored"]) > 0:
            s = row["region_start"]
            e = row["region_end"]
            previously_scored_regions[s:e+1] = 1.0

# now let's modify track quality s.t. swaps register as 1.0's [if not previously scored + rescan]
diff = (iou > alt_iou) * 1
flipLocs = np.where(1*(diff[1:] != diff[:-1]))[0]

for fl in flipLocs:
    if previously_scored_regions[fl] == 0:
        min_fl = max(0, fl-4)
        max_fl = min(iou.shape[0], fl+4)
        candidate_search_quality[min_fl:max_fl] = 1.0

# any previously scored section doesn't need to be rescored.
candidate_search_quality[previously_scored_regions > 0] = 0

labeled_objects, _ = label_objects(candidate_search_quality == 1.0)
swapRes = [int((k[0].start + k[0].stop)/2) for k in find_objects(labeled_objects)]
swapLocs = (list(zip(swapRes, candidate_search_quality[swapRes])))

allRes, _ = find_peaks(candidate_search_quality, distance=int(region_size/2))
worstLocs = (list(zip(allRes, candidate_search_quality[allRes])))
worstLocs = list(filter(lambda x: x[1] > QUALITY_THRESHOLD, worstLocs))
worstLocs = sorted(worstLocs, key=lambda x: x[1], reverse=True)
worstLocs = list(filter(lambda x: x not in swapLocs, worstLocs))

# show a couple of graphs to provide some context on tracking
fig = plt.figure(figsize=(6,4))
plt.plot(np.arange(track_quality.shape[0]), 1-track_quality, color="g")
plt.hlines(1-QUALITY_THRESHOLD, xmin=0, xmax=track_quality.shape[0])
plt.title(f"ZOD-Quality Preview (#ZOD: {len(worstLocs)}. Threshold = {1-QUALITY_THRESHOLD})")
plt.ylim(0, 1.05)
plt.show(block=False)

for mode, locs in [["SWAPS", swapLocs], ["WORSTLOCS", worstLocs]]:
    nLocs = len(locs)
    print(f"To process: {mode} [{nLocs} locs]")

fig_local = ""
control_seeked = False
control_dirtyplot = False
control_figure_section_selector = "LEFT"
control_window_mode = "ZOD"

def redraw_local_plot(thiszod, section_start, section_end, preview_start, preview_end):
    global fig_local
    nevershown = False
    loc = section_start + np.argmax(track_quality[section_start:section_end+1])
    locquality = track_quality[loc]

    print(f"All figures before: {plt.get_fignums()}")
    print(f"Current figure before: {plt.gcf().number if plt.get_fignums() else 'None'}")

    # show a couple of graphs to provide some context on tracking

    if fig_local == "" or not plt.fignum_exists(fig_local.number):
        fig_local = plt.figure(figsize=(6,4))
        cid = fig_local.canvas.mpl_connect('button_press_event', local_plot_onclick)
        nevershown = True

    fig = plt.figure(fig_local)
    plt.clf()
    ax1, ax2 = fig.subplots(nrows=2)
    plt.suptitle(f"ZOD-{mode} [Local] Q={1-locquality:0.02} @ F:{loc} W:{section_end}-{section_start}")
    ax1.set_title(f"Click to set {control_window_mode} {control_figure_section_selector} boundary.")

    #OLD COLOR
    #plt.plot(np.arange(track_quality.shape[0]), 1-track_quality, color="g", alpha=0.3)
    ax2.plot(np.arange(track_quality.shape[0]), 1-track_quality, color="blue", alpha=1.0)

    #OLD COLOR
    #plt.plot(np.arange(iou.shape[0]), iou, color="coral", alpha=0.4)
    ax1.plot(np.arange(iou.shape[0]), iou, color="#367E7F", alpha=0.8)

    #OLD COLOR
    #plt.plot(np.arange(alt_iou.shape[0]), alt_iou, color="b", alpha=0.4)
    ax1.plot(np.arange(alt_iou.shape[0]), alt_iou, color="#D95D27", alpha=0.8)

    ax1.hlines(1-QUALITY_THRESHOLD, xmin=0, xmax=track_quality.shape[0])
    ax1.vlines(preview_start, 0, 1.0, linestyles="dotted")
    ax2.vlines(preview_start, 0, 1.0, linestyles="dotted")
    ax1.vlines(preview_end, 0, 1.0, linestyles="dotted")
    ax2.vlines(preview_end, 0, 1.0, linestyles="dotted")

    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax1.set_xticks([])
    ax1.set_ylabel("IOUs")
    ax2.set_ylabel("Quality")

    inds = np.arange(track_quality.shape[0])
    in_section = np.logical_and(inds >= section_start, inds <= section_end)
    #OLD COLOR
    #plt.fill_between(np.arange(track_quality.shape[0]), in_section, step="mid", alpha=0.2, color="b")
    ax1.fill_between(np.arange(track_quality.shape[0]), in_section, step="mid", alpha=0.5, color="#FFB61E")
    ax2.fill_between(np.arange(track_quality.shape[0]), in_section, step="mid", alpha=0.5, color="#FFB61E")
    d = preview_end - preview_start
    center = (preview_end + preview_start) / 2
    l_lim = max(0, center - d)
    r_lim = min(track_quality.shape[0], center + d)
    if r_lim == l_lim:
        l_lim -= 5
        r_lim += 5
    l_lim = max(0, l_lim)
    r_lim = min(video_length - 1, r_lim)
    ax1.set_xlim(l_lim, r_lim)
    ax2.set_xlim(l_lim, r_lim)

    # draw any relevant other sections:
    other_swaps = np.zeros(track_quality.shape[0])
    other_worstlocs = np.zeros(track_quality.shape[0])
    other_scored_swaps = np.zeros(track_quality.shape[0])
    other_scored_worstlocs = np.zeros(track_quality.shape[0])
    for zod in zod_inventory:
        zstart = zod["region_start"]
        zend = zod["region_end"]

        # don't double_draw this one
        if (zstart == section_start and zend == section_end) or \
                thiszod["zidx"] == zod["zidx"]:
            continue

        # skip anything we're not gonna see anyway
        if zstart > r_lim or zend < l_lim:
            continue
        zscored = zod["previously_scored"]
        zsource = zod["source"]

        if zstart > 342000 and zstart < 344000:
            print(f"doing")
            print(thiszod)
            print(zod)
        if zsource == "WORSTLOCS":
            if zscored:
                other_scored_worstlocs = np.logical_or(other_scored_worstlocs,
                                                       np.logical_and(inds >= zstart, inds <= zend))
            else:
                other_worstlocs = np.logical_or(other_worstlocs,
                                                np.logical_and(inds >= zstart, inds <= zend))
        else:  # zsrouce = SWAPS or other
            if zscored:
                other_scored_swaps = np.logical_or(other_scored_swaps,
                                            np.logical_and(inds >= zstart, inds <= zend))
            else:
                other_swaps = np.logical_or(other_swaps,
                                            np.logical_and(inds >= zstart, inds <= zend))
    #ax1.fill_between(inds, other_swaps, step="mid", alpha=0.5, color="tab:olive", hatch='X')
    #ax1.fill_between(inds, other_scored_swaps, step="mid", alpha=0.5, color="tab:olive", hatch='///')
    #ax2.fill_between(inds, other_swaps, step="mid", alpha=0.5, color="tab:olive", hatch='X')
    #ax2.fill_between(inds, other_scored_swaps, step="mid", alpha=0.5, color="tab:olive", hatch='///')

    #ax1.fill_between(inds, other_worstlocs, step="mid", alpha=0.5, color="tab:red", hatch='X')
    #ax1.fill_between(inds, other_scored_worstlocs, step="mid", alpha=0.5, color="tab:red", hatch='///')
    #ax2.fill_between(inds, other_worstlocs, step="mid", alpha=0.5, color="tab:red", hatch='X')
    #ax2.fill_between(inds, other_scored_worstlocs, step="mid", alpha=0.5, color="tab:red", hatch='///')

    fig.canvas.draw()
    if nevershown:
        plt.show(block=False)
        nevershown = False

def local_plot_onclick(event):
    if event.name != 'button_press_event' or event.button != 1 or \
           type(event.xdata) == type(None):
        return
    update_boundaries(event.xdata)

def update_boundaries(xpos):
    global control_figure_section_selector
    global control_seeked
    global control_dirtyplot
    global preview_start
    global preview_end
    global section_start
    global section_end
    global control_window_mode

    if control_window_mode == "PREVIEW":
        if control_figure_section_selector == "LEFT":
            control_figure_section_selector = "RIGHT"

            # does this make sense within the current ZOD?
            if (preview_end - xpos) < 2 or (section_start - xpos) < 2:
                print("Rejecting invalid start boundary.")
                control_dirtyplot = True
                return

            preview_start = max(0, int(xpos))
            frame_num = preview_start
            control_seeked = True
            control_dirtyplot = True
        elif control_figure_section_selector == "RIGHT":
            control_figure_section_selector = "LEFT"
            if (xpos - section_end) < 2 or (xpos - preview_start) < 2:
                print("Rejecting invalid end boundary.")
                control_dirtyplot = True
                return
            preview_end = min(video_length - 1, int(xpos))
            frame_num = preview_start
            control_seeked = True
            control_dirtyplot = True
        print(f"New window: {section_start}-{section_end} preview: {preview_start}-{preview_end}")
    elif control_window_mode == "ZOD":
        if control_figure_section_selector == "LEFT":
            control_figure_section_selector = "RIGHT"
            t_start = max(0, int(xpos))
            if (section_end - t_start) < 1:
                print("Rejecting invalid start boundary.")
                control_dirtyplot = True
                return

            # does this conflict with any existing, scored ZOD?
            for zt in zod_inventory:
                if not zt["previously_scored"]:
                    continue
                if (zt["region_start"] >= t_start and zt["region_start"] <= section_end) or \
                   (zt["region_end"] >= t_start and zt["region_end"] <= section_end):
                    print("Rejecting invalid start boundary (overlap scored ZOD)")
                    control_dirtyplot = True
                    return

            section_start = t_start
            preview_start = max(0, t_start - (section_end - section_start))

            frame_num = preview_start
            control_seeked = True
            control_dirtyplot = True
        elif control_figure_section_selector == "RIGHT":
            control_figure_section_selector = "LEFT"
            t_end = min(video_length - 1, int(xpos))

            if (t_end - section_start) < 1:
                print("Rejecting invalid end boundary.")
                control_dirtyplot = True
                return

            # does this conflict with any existing, scored ZOD?
            for zt in zod_inventory:
                if not zt["previously_scored"]:
                    continue
                if (zt["region_start"] >= section_start and zt["region_start"] <= t_end) or \
                   (zt["region_end"] >= section_start and zt["region_end"] <= t_end):
                    print("Rejecting invalid start boundary (overlap scored ZOD)")
                    control_dirtyplot = True
                    return

            section_end = t_end
            preview_end = min(video_length - 1, t_end + (section_end - section_start))

            control_dirtyplot = True
        print(f"New window: {section_start}-{section_end} preview: {preview_start}-{preview_end}")

def zoomIn():
    global control_seeked
    global control_dirtyplot
    global preview_start
    global preview_end
    global section_start
    global section_end

    # halve the distance between section_boundary/preview_boundary
    delta_left = max(int((section_start - preview_start)/2), 50)
    delta_right = max(int((preview_end - section_end)/2), 50)
    preview_start = preview_start + delta_left
    preview_end = preview_end - delta_right
    control_dirtyplot = True

def zoomOut():
    global control_seeked
    global control_dirtyplot
    global preview_start
    global preview_end
    global section_start
    global section_end

    # double the distance between section_boundary/preview_boundary, minimum 50
    delta_left = max(section_start - preview_start, 50)
    delta_right = max(preview_end - section_end, 50)
    preview_start = max(preview_start - delta_left, 0)
    preview_end = min(preview_end + delta_right, video_length - 1)

    control_dirtyplot = True

zod_inventory = []
def update_zod_inventory():
    global zod_inventory
    zod_inventory = []

    # load the csv first
    df = load_zod_csv(exp)
    for zi in range(len(df)):
        zs, ze, zsource, zq = df.loc[zi, ["region_start", "region_end", "source", "best_cutie_quality"]]
        zod_inventory.append({
            "region_start": zs,
            "region_end": ze,
            "previously_scored": 1,
            "source": zsource,
            "zidx": zi,
        })
    print(f"Prior inventory:")
    print(zod_inventory)
    # add back in any locs not yet scored
    for mode, locs in [["SWAPS", swapLocs], ["WORSTLOCS", worstLocs]]:
        for loc, q in locs:
            # if this loc is inside the inventory, update it if it's missing metadata, then skip it.

            found = False
            for zod in zod_inventory:
                if loc >= zod["region_start"] and loc <= zod["region_end"]:
                    if zod["source"] == "UNKNOWN":
                        zod["source"] = mode
                    found = True

            if found == True:
                print(f"in zod-inventory update: {loc} is already captured, skipping")
                continue

            print(f"Evaluating {loc}")

            # otherwise, estimate the region parameters - prune any neighboring zones
            r_start_orig = max(0, loc - int(region_size/2))
            r_end_orig = min(loc + int(region_size/2), iou.shape[0])
            region_template = np.zeros_like(previously_scored_regions)
            region_template[max(0, loc - int(region_size/2)):min(loc + int(region_size/2)+1, iou.shape[0])] = 1
            region_template[previously_scored_regions > 0] = 0

            # next, peak_widths!
            if region_template[loc] == 0:
                print(f"Loc {loc} was in a previously scored ZOD.")
                continue
            _, heights, lbound, rbound = peak_widths(region_template, [loc])
            r_start = int(np.ceil(lbound[0]))
            r_end = int(np.floor(rbound[0]))
            if r_end == r_start and heights[0] == 0:
                print("Pruned a location that had already been scored.")
                continue
            if r_end == r_start and heights[0] > 0:
                print(f"******** There is a 1 frame ZOD at {r_start}!")
                r_start = max(0, r_start - 2)
                r_end = min(video_length - 1, r_end + 2)

            if r_start_orig != r_start or r_end_orig != r_end:
                print(f"after early neighbor pruning, {r_start_orig}:{r_end_orig} -> {r_start}:{r_end}")

            if not rescan_mode:
                # let's try narrowing these down a little bit - narrow window until peaks > 1
                q_tmp = np.zeros_like(track_quality)
                q_tmp[r_start:r_end] = track_quality[r_start:r_end]

                # flag transitions from acceptable to unacceptable quality for window-finding
                q_tmp[r_start:r_end][q_tmp[r_start:r_end] < QUALITY_THRESHOLD] = 0.0

                # scan with smaller windows in order to identify where the badness is most concentrated.
                # regions that remain at a wide gap may still be later investigated.
                last_d = int(region_size * .666)
                for d in np.linspace(int(region_size * .666), 5).astype(int):
                    if mode == "SWAPS":
                        tres, _ = find_peaks(q_tmp * (q_tmp > 1.0), distance=d)
                    else:
                        tres, _ = find_peaks(q_tmp, distance=d)
                    wtres = (list(zip(tres, q_tmp[tres])))
                    wtres = list(filter(lambda x: x[1] > QUALITY_THRESHOLD, wtres))
                    if len(wtres) > 1:
                        break
                    last_d = d

                r_start_orig = max(0, loc - last_d)
                r_end_end = min(iou.shape[0]-1, loc + last_d)

                # prune any neighboring zones
                region_template = np.zeros_like(previously_scored_regions)
                region_template[r_start_orig:r_end_orig+1] = 1
                region_template[previously_scored_regions > 0] = 0

                if region_template[loc] == 0:
                    print(f"Loc {loc} was in a previously scored ZOD.")
                    continue
                _, _, lbound, rbound = peak_widths(region_template, [loc])
                r_start = int(np.ceil(lbound[0]))
                r_end = int(np.floor(rbound[0]))

                if r_start_orig != r_start or r_end_orig != r_end:
                    print(f"after post-d neighbor pruning, {r_start_orig}:{r_end_orig} -> {r_start}:{r_end}")

            zod_inventory.append({
                "region_start": r_start,
                "region_end": r_end,
                "previously_scored": 0,
                "source": mode,
                "zidx": len(zod_inventory),
            })

update_zod_inventory()

for mode, locs in [["SWAPS", swapLocs], ["WORSTLOCS", worstLocs]]:
    nLocs = len(locs)
    print(f"Processing {mode} [{nLocs} locs]")
    print(f"Locations: {locs}")
    for wlidx in range(len(locs)):
        loc, locquality = locs[wlidx]
        found = False
        for zidx in range(len(zod_inventory)):
            zod = zod_inventory[zidx]
            if loc >= zod["region_start"] and loc <= zod["region_end"]:
                found = True
                break
        if not found:
            print(f"*** Inconsistent ZOD inventory - did not find {loc}!")
            sys.exit(1)

        if zod["previously_scored"]:
            print(f"Skipping previously scored {loc}")
            continue

        print(f"Considering ZOD {mode} circa frame {loc} [Score: {locquality:0.02}].  Loc# {wlidx+1} / {len(locs)}")

        r_start = zod["region_start"]
        r_end = zod["region_end"]
        d = r_end - r_start
        buffer = min(d, 100)

        redraw_local_plot(zod, r_start, r_end, r_start-d, r_end+d)

        section_start = r_start
        section_end = r_end
        preview_start = max(section_start - buffer, 0)
        preview_end = min(section_end + buffer, video_length - 1)
        ok, results = navigateSection(exp, zod, mp4_in, masks_fwd_path, masks_rev_path, sleap_path)
        if not ok:
            print(f"Skipped [{r_start},{r_end}]")
            continue

        print(f"Saving ZOD annotations for [{mode}] {r_start}-{r_end}")
        print(f"Success: {ok}")
        print(f"Results: {results}")
        df = load_zod_csv(exp)
        print("Loading zod inventory prior to saving with an update")
        print(df)
        df_len = len(df.index)
        df.loc[df_len, "region_start"] = section_start
        df.loc[df_len, "region_end"] = section_end
        df.loc[df_len, "previously_scored"] = 1
        df.loc[df_len, "source"] = mode
        df.loc[df_len, "cutie_forward_quality"] = results["cutie_forward_quality"]
        df.loc[df_len, "cutie_forward_swapped"] = results["cutie_forward_swapped"]
        df.loc[df_len, "cutie_reverse_quality"] = results["cutie_reverse_quality"]
        df.loc[df_len, "cutie_reverse_swapped"] = results["cutie_reverse_swapped"]
        df.loc[df_len, "best_cutie"] = results["best_cutie"]
        df.loc[df_len, "best_cutie_swapped"] = results["best_cutie_swapped"]
        df.loc[df_len, "best_cutie_quality"] = results["best_cutie_quality"]
        df.loc[df_len, "sleap_raw_quality"] = results["sleap_raw_quality"]
        df.loc[df_len, "sleap_raw_swapped"] = results["sleap_raw_swapped"]
        df.loc[df_len, "proofed"] = ""
        df["region_start"] = df["region_start"].astype(int)
        df["region_end"] = df["region_end"].astype(int)
        df["previously_scored"] = df["previously_scored"].astype(int)
        if "sleap_interpolated_quality" in results:
            df.loc[df_len, "sleap_interpolated_quality"] = results["sleap_interpolated_quality"]
        print("Updated inventory:")
        print(df)

        if rescan_mode:
            df.to_csv(dataframe_for_results_rescan, index=False)
        else:
            df.to_csv(dataframe_for_results, index=False)
