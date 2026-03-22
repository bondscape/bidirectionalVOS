# bidirectionalVOS

# Welcome!

(This guide is still a work in progress)

# Setup Instructions

We recommend creating a dedicated conda environment for this project.
You can find conda here: https://www.anaconda.com/download

This workflow changes a bit if your GPU is newer (RTX 50-series+) vs older. To test which:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

If the number comes back < 12.0, try the following:
```bash
conda create --name vos_pipeline python=3.10
conda activate vos_pipeline
conda install ffmpeg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/hkchengrex/Cutie.git
python -m cutie.utils.download_models
pip install opencv-python tk runpod matplotlib
```

If the number comes back >= 12.0, try the following:
```bash
conda create --name vos_pipeline python=3.12
conda activate vos_pipeline
conda install ffmpeg
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/hkchengrex/Cutie.git --no-deps
pip install \
  cython \
  'gitpython>=3.1' \
  'thinplate@git+https://github.com/cheind/py-thin-plate-spline' \
  'hickle>=5.0' \
  'tensorboard>=2.11' \
  'numpy>=1.21' \
  'Pillow>=9.5' \
  'opencv-python>=4.8' \
  'scipy>=1.7' \
  'pycocotools>=2.0.7' \
  'tqdm>=4.66.1' \
  'gradio>=3.34' \
  'gdown>=4.7.1' \
  'einops>=0.6' \
  'hydra-core>=1.3.2' \
  'charset-normalizer>=3.1.0' \
  'netifaces>=0.11.0' \
  'easydict' \
  'av>=0.5.2' \
  'requests'

python -m cutie.utils.download_models
pip install opencv-python tk runpod matplotlib
```

Either way, when you're done setting up, you can confirm torch has CUDA support with:
```bash
python -c "import torch; print(torch.cuda.get_device_capability()); torch.zeros(1).cuda(); print('Torch w/ CUDA support is working')"
```


# Source contents:
```
    pipeline/         # Start here - contains tools for generating candidate keyframes,
                      #     sets up for running bidirectional cutie inference, assembles results

    segmenter/        # standalone or runpod-instanced tool for creating keyframe masks

    cutie_inference/  # invokes Cutie in a direction over a given video clip with reference frames.

    cyoa/             # interactive tool to identify, define, and describe Zones of Disagreement output
                      #     from bidirectional Cutie inference

    example_project/  # an example small dataset
```

# Try it out on the example project:

We've included a small clip from two videos of opposite sex voles to allow for convenient exploration with
this toolset, located in the example_project directory.  Several steps of the processing pipeline are completed
for you in advance for your convenience, though they can be safely repeated or explored.

Steps:

Change to the pipeline folder:
```bash
cd pipeline
```

Thumbnail generation (can be skipped for the example project)
```bash
python generate_thumbnails.py --video ../example_project/videos/e06-C-Intro-MF1-clip.mp4 --output_path ../example_project/thumbnails/e06-C-Intro-MF1-clip --frames_to_output 50
python generate_thumbnails.py --video ../example_project/videos/e06-C-Intro-MF2-clip.mp4 --output_path ../example_project/thumbnails/e06-C-Intro-MF2-clip --frames_to_output 50
```

Interactive machine-assisted mask generation (can be skipped for the example project)
```bash
python ../segmenter/segmenter-client.py --input_path ../example_project/thumbnails/e06-C-Intro-MF1-clip
python ../segmenter/segmenter-client.py --input_path ../example_project/thumbnails/e06-C-Intro-MF2-clip
python ../segmenter/segmenter-proof.py --input_path ../example_project/thumbnails/e06-C-Intro-MF1-clip
python ../segmenter/segmenter-proof.py --input_path ../example_project/thumbnails/e06-C-Intro-MF2-clip
```

Scheduling the bidirectional video object segmentation - outputs to inference_tasks.txt)
```bash
python schedule_cutie_inference.py --video ../example_project/videos/e06-C-Intro-MF1-clip.mp4 --thumbnail_path ../example_project/thumbnails/e06-C-Intro-MF1-clip/proofed --output_path ../example_project/cutie_inference/e06-C-Intro-MF1-clip
python schedule_cutie_inference.py --video ../example_project/videos/e06-C-Intro-MF2-clip.mp4 --thumbnail_path ../example_project/thumbnails/e06-C-Intro-MF2-clip/proofed --output_path ../example_project/cutie_inference/e06-C-Intro-MF2-clip
```

Actually execute the tasks in inference_tasks.txt - will take a while!

After all cutie inference is done, combine segments together:
```bash
python pc_stitching.py --segments_path ../example_project/cutie_inference/e06-C-Intro-MF1-clip/cutie/
python pc_stitching.py --segments_path ../example_project/cutie_inference/e06-C-Intro-MF2-clip/cutie/
```

Using the CYOA tools to resolve the disagreements between video segmentations and align the SLEAP and video segmentations:
```
cd ../cyoa
python animal_labeler.py --video_path ../example_project/videos/e06-C-Intro-MF1-clip.mp4
python cyoa_tool.py --video_path ../example_project/videos/e06-C-Intro-MF1-clip.mp4 --cutie_path  ../example_project/cutie_inference/e06-C-Intro-MF1-clip/cutie --sleap_path ../example_project/sleap/tracked/e06-C-Intro-MF1.labels.0.tracked.cleaned.h5
python align_sleap_and_cutie.py --video_path ../example_project/videos/e06-C-Intro-MF1-clip.mp4 --cutie_path  ../example_project/cutie_inference/e06-C-Intro-MF1-clip/cutie --sleap_path ../example_project/sleap/tracked/e06-C-Intro-MF1.labels.0.tracked.cleaned.h5 --output_path ./output/e06-C-Intro-MF1-clip --parts_config parts_config.json --identities_config_path ../example_project/videos/e06-C-Intro-MF1-clip.mp4.identities.json --annotations e06-C-Intro-MF1-clip.annotations.csv --do_video
```


