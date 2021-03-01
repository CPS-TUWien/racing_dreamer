.PHONY: build deploy
ARC_GYM_BRANCH=master
TAG=latest
MODEL=models/

allagents := $(shell ls agents)
allbuilds := $(addprefix build-,$(allagents))
allruns   := $(addprefix run-,$(allagents))
allbashs  := $(addprefix bash-,$(allagents))

AGENT_SELECT?=dreamer
AGENT_CHECKPOINT?=treitlstrasse_dreamer

$(allbuilds):
	$(eval AGENT_NAME := $(shell echo "$@" | cut -d "-" -f 2))
	@echo "AGENT: \"$(AGENT_NAME)\""
	@echo "TAG:   \"$(TAG)\""
	@echo "MODEL: \"$(MODEL)\""
	rm -rf Dockerfile
	test -f agents/$(AGENT_NAME)/Dockerfile.include && cat Dockerfile.head agents/$(AGENT_NAME)/Dockerfile.include Dockerfile.tail > Dockerfile || echo "no Dockerfile.include for agent"
	test -f agents/$(AGENT_NAME)/Dockerfile && cat agents/$(AGENT_NAME)/Dockerfile > Dockerfile || echo "no Dockerfile for agent"
	test -f agents/$(AGENT_NAME)/Dockerfile.car && uname -a | grep -q "tegra " && cat agents/$(AGENT_NAME)/Dockerfile.car > Dockerfile || echo "not running on car hardware"
	rm -rf tmp_model_dir_within_build_context
	mkdir -p tmp_model_dir_within_build_context
	cp -ar $(MODEL) tmp_model_dir_within_build_context
	docker build -t tuwcps/arc_agent_ros:$(AGENT_NAME) --build-arg MODEL=tmp_model_dir_within_build_context --build-arg GYMDIR=tmp_model_dir_within_build_context --build-arg AGENT=$(AGENT_NAME) .;
	rm -rf tmp_model_dir_within_build_context

$(allbashs):
	$(eval AGENT_NAME := $(shell echo "$@" | cut -d "-" -f 2))
	@echo "(make) AGENT_NAME: \"$(AGENT_NAME)\""
	@echo "(make) AGENT_SELECT: \"$(AGENT_SELECT)\""
	@echo "(make) AGENT_CHECKPOINT: \"$(AGENT_CHECKPOINT)\""
	docker run -it --name=f1tenth_agent_$(TAG)_bash_container --rm --net=host -v $(shell pwd):/cwd \
	--env AGENT_NAME=$(AGENT_NAME) \
	--env AGENT_SELECT=$(AGENT_SELECT) \
	--env AGENT_CHECKPOINT=$(AGENT_CHECKPOINT) \
	tuwcps/arc_agent_ros:$(AGENT_NAME) "/bin/bash"

$(allruns):
	$(eval AGENT_NAME := $(shell echo "$@" | cut -d "-" -f 2))
	@echo "(make) AGENT_NAME: \"$(AGENT_NAME)\""
	@echo "(make) AGENT_SELECT: \"$(AGENT_SELECT)\""
	@echo "(make) AGENT_CHECKPOINT: \"$(AGENT_CHECKPOINT)\""
	docker run -it --name=f1tenth_agent_$(AGENT_NAME)_container --rm --net=host -v $(shell pwd):/cwd \
	--env AGENT_NAME=$(AGENT_NAME) \
	--env AGENT_SELECT=$(AGENT_SELECT) \
	--env AGENT_CHECKPOINT=$(AGENT_CHECKPOINT) \
	tuwcps/arc_agent_ros:$(AGENT_NAME)

video-demo:
	docker run -it --name=f1tenth_agent_$(TAG)_bash_container --rm --net=host -v $(shell pwd):/cwd \
	--env AGENT_NAME=$(AGENT_NAME) \
	tuwcps/arc_agent_ros:no_ros_make_video "/bin/bash" "-c" \
		"/usr/bin/python3 /ws/src/f1tenth_agent_ros/src/agent.py \
		--task_name SingleAgentColumbia-v0 \
		--time_limit 30.0 \
		--store_video"

run: run-ego

docker-purge-all:
	docker container rm $(shell docker container ls -a -q) # remove all stopped containers
	docker rmi $(shell docker images -a -q) # remove all images

docker-status:
	docker image ls


