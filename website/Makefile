tag=latest
organization=koldev45
distant_image=websites-repo
local_image=bstl_website

build:
	docker build --force-rm $(options) -t bstl_website:latest .

build-prod:
	$(MAKE) build options="--target production"

push:
	docker tag $(local_image):latest $(organization)/$(distant_image):$(tag)
	docker push $(organization)/$(distant_image):$(tag)

compose-start:
	docker-compose up --remove-orphans $(options)
