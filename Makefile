DEPLOY_DIR=deploy
IMAGE_NAME=dc_deploy
CONTAINER_NAME=deploy
build:
	docker build -t $(IMAGE_NAME) $(DEPLOY_DIR)

run:
	docker run -d --name $(CONTAINER_NAME) -p 5001:5001 $(IMAGE_NAME)
stop:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

clean:
	docker rmi $(IMAGE_NAME)

all: build run

