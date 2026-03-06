This is the segmenter server, which is powered by Segment Anything Model 2 (SAM 2) and runs on runpod.io
as a "serverless" endpoint.

Our runpod.io serverless endpoint configuration data is below:

    GPU: 16-24GB
    Max Workers: 1-2
    GPU Count: 1
    Idle Timout: 1800 seconds
    Enable Flashboot: yes

    environmental variables
        MODE : cuda

    docker image (the below may work for you, or you may need to build your own using the build script here)
        kquine/projects:segmenter-server
        dockerhub

If you would like to build your own docker image:
    1) download the model weights for the vit_h model of SAM 2:
        As of 2026-January-25:
            https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

    2) update Dockerfile to reference the sam_vit_h_*.ph if different than above,
    3) build and push the docker image:
            docker build . --tag YOUR_DOCKER_USERID:segmenter-server --platform linux/amd64
            docker tag YOUR_DOCKER_USERID:segmenter-server YOUR_DOCKER_USERID/projects:segmenter-server
            docker push YOUR_DOCKER_USERID:segmenter-server
