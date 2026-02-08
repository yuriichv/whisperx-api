# Makefile for building and managing Docker container for whisperx-api

IMAGE_NAME ?= whisperx-api
TAG ?= v0.0.1
CONTAINER_NAME ?= whisperx-api

# Build Docker image
build:
	docker build -t $(IMAGE_NAME):$(TAG) .

# Run container
run:
	docker run -d --name $(CONTAINER_NAME) -p 8000:8000 $(IMAGE_NAME):$(TAG)

# Stop container
stop:
	docker stop $(CONTAINER_NAME) && docker rm $(CONTAINER_NAME)

# View logs
logs:
	docker logs -f $(CONTAINER_NAME)

# Rebuild and restart container
restart: stop build run

.PHONY: build run stop logs restart
