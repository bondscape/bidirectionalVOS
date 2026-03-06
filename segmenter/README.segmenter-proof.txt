This is the client portion of a tool to assist with mask generation using segment-anything.

You will need to generate a config file reflecting your runpod configuration information.
    runpod.config.json

When the tool starts up it will pop up a chooser - it is asking you to select a folder (not the frames themselves).

That folder should contain inputs that look like:
    0000000.jpg
    0001554.jpg
    0003109.jpg
    0004664.jpg
        (etc)

The numeric portion of these images represents the frame number in the corresponding video.

Segmented output images will be placed in a subdirectory called "segmented".

After selecting the input folder, there will be a brief pause while connecting to your runpod instance.


================
    Controls
================

    Escape      quit

    Spacebar:   next candidate image
    p           prev candidate image

    o           change currently selected mask / brush color
    click       place a marker for mask selection
    s           generate segment-anything masks. "in" points are currently selected color. non-matching colors are "out"
    d           switch between the provided segment-anything masks.
    '1' or '2'  assign currently selected SA mask as animal ID 1 or 2.  OK to designate multiple masks as 1 or 2.
    r           reset current working masks image (followed by 'w' to clear any saved copy)
    w           save current working masks image in output folder.

    Other keys that exist but are less useful:
    =           seek to a random frame (not useful)
    /           take screenshot

============
    Tips
============
   * make sure frame 0 and the final frame are both labeled ('p' from frame zero gets you last frame)
   * How large should unlabeled intervals be?  We found 5k-20k frames long segments to be a good number.
   * try to pick frames where the animals are separated or at least have clear boundaries.
   * skip frames where masks generate badly despite a couple of attempts. Good labels seems better than more labels.
   * Should you include tails? We allowed tails but didn't aim to include them in masks.
   * Use the third color to exclude things you don't want to include - ports, tails (if problematic), etc.
   * you can select multiple parts of an animal serially - save will consolidate multiple '1' and '2' segments.

