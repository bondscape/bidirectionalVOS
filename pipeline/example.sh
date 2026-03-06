# first, lets make some thumbnails for our example videos... (if not already done)
#python generate_thumbnails.py --video ../example_project/videos/e06-C-Intro-MF1-clip.mp4 --output_path ../example_project/thumbnails/e06-C-Intro-MF1-clip --frames_to_output 50
#python generate_thumbnails.py --video ../example_project/videos/e06-C-Intro-MF2-clip.mp4 --output_path ../example_project/thumbnails/e06-C-Intro-MF2-clip --frames_to_output 50

# next, we should generate masks (interactive)
#python ../segmenter/segmenter-client.py --input_path ../example_project/thumbnails/e06-C-Intro-MF1-clip
#python ../segmenter/segmenter-client.py --input_path ../example_project/thumbnails/e06-C-Intro-MF2-clip
#python ../segmenter/segmenter-proof.py --input_path ../example_project/thumbnails/e06-C-Intro-MF1-clip
#python ../segmenter/segmenter-proof.py --input_path ../example_project/thumbnails/e06-C-Intro-MF2-clip

# next, we schedule cutie inference (appends commands to run / schedule to inference_tasks.txt)
#python schedule_cutie_inference.py --video ../example_project/videos/e06-C-Intro-MF1-clip.mp4 --thumbnail_path ../example_project/thumbnails/e06-C-Intro-MF1-clip/proofed --output_path ../example_project/cutie_inference/e06-C-Intro-MF1-clip
#python schedule_cutie_inference.py --video ../example_project/videos/e06-C-Intro-MF2-clip.mp4 --thumbnail_path ../example_project/thumbnails/e06-C-Intro-MF2-clip/proofed --output_path ../example_project/cutie_inference/e06-C-Intro-MF2-clip

# after cutie inference is done, we have to combine all the cutie segments together:
#python pc_stitching.py --segments_path ../example_project/cutie_inference/e06-C-Intro-MF1-clip/cutie/
#python pc_stitching.py --segments_path ../example_project/cutie_inference/e06-C-Intro-MF2-clip/cutie/
