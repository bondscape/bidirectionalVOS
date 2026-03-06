docker build . --tag YOUR_DOCKER_ID:segmenter-server --platform linux/amd64 --no-cache-filter=8
docker tag YOUR_DOCKER_ID:segmenter-server YOUR_DOCKER_ID/projects:segmenter-server
docker push YOUR_DOCKER_ID/projects:segmenter-server
