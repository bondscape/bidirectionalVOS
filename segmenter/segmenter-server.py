import runpod
import base64
import argparse
import numpy as np
import time
import threading
import os
import sys

from segment_anything import SamPredictor, sam_model_registry

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="", help="specify cuda or cpu mode")
args = parser.parse_args()

devoption = ""
if args.mode != "":
    devoption = args.mode
    print(f"setting mode from commandline: {devoption}")
elif os.environ.get("MODE") != None:
    devoption = os.environ["MODE"]
    print(f"setting mode from environment: {devoption}")
if devoption == "":
    devoption = "cuda"
    print(f"setting mode from default: {devoption}")
if devoption not in ["cpu", "cuda"]:
    print(f"Unrecognized mode option: {devoption} - valid options: cuda, cpu.")
    print(parser.print_help(sys.stderr))
    sys.exit(1)

modelname = "sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=modelname)
sam.to(device=devoption)
predictor = SamPredictor(sam)

start_time = time.time()
cache = {}
cachelock = threading.Lock()
MAX_CACHE = 2

def processImageWithCache(image, checksum):
    global cache
    global cachelock

    cachelock.acquire()
    if checksum in cache:
        print(f"Using cache entry for {checksum}")
        cache[checksum]["last"] = time.time()
        predictor = cache[checksum]["predictor"]
        cachelock.release()
        return predictor

    # if we are out of room room in the cache, evict the oldest entry.
    while len(cache.keys()) > MAX_CACHE:
        # find the oldest, evict it.
        oldest = float('inf')
        oldest_k = None
        for k in cache.keys():
            if cache[k]["last"] < oldest:
                oldest = cache[k]["last"]
                oldest_k = k
        print(f"Dropping cache entry {oldest_k} from cache.")
        del cache[k]

    # add to the cache!
    print(f"Adding new item {checksum} to cache!")
    cache[checksum] = {}
    cache[checksum]["last"] = time.time()

    predictor = SamPredictor(sam)
    predictor.set_image(image)
    cache[checksum]["predictor"] = predictor
    cachelock.release()

    return predictor

async def handler(job):
    global lastDigestedImageChecksum
    print(f"Got request: {job['id']}")
    input = job["input"]
    if "verb" not in input or "health" == input["verb"]:
        cur_time = time.time()
        return {
            "status": "online",
            "uptime": cur_time - start_time,
        }

    if input["verb"] == "predict":
        # unmarshal image
        colorframe = np.frombuffer(base64.b64decode(input["colorframe"]["data"]), dtype=np.uint8)
        colorframe = colorframe.reshape(input["colorframe"]["height"], input["colorframe"]["width"], 3)

        thisChecksum = np.sum(colorframe)
        predictor = processImageWithCache(colorframe, thisChecksum)

        samPoints = input["samPoints"]
        point_labels = input["point_labels"]
        print("Predicting.")
        samOutputMasks, _, _ = predictor.predict(point_coords = np.array(samPoints),
                                                 point_labels = point_labels)
        print("Request complete.")
        return {
          "samOutputMasks": {
                "count": samOutputMasks.shape[0],
                "height": samOutputMasks.shape[1],
                "width": samOutputMasks.shape[2],
                "data": base64.b64encode(samOutputMasks.astype(np.uint8)).decode('ascii'),
          }
        }
    else:
        errstr = f"unrecognized verb: {input['verb']}"
        return {
            "error": errstr,
        }

print("segmenter-server: executing runpod handler.")
runpod.serverless.start({ "handler": handler})
