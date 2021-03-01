.PHONY: build deploy

allmaps := $(shell "ls ../maps | grep yaml")
allbuilds := $(addprefix build-,$(allmaps))

$(allbuilds):
	echo "need to build: $@"

all: \
output/plechaty1.npz \
output/berlin.npz \
output/skirk.npz \
output/mtl.npz \
output/columbia.npz \
output/torino_redraw_small.npz \

clean:
	rm -rf output

%: output/%.npz
	echo "generated $@"

output/%.npz: ../maps/%.yaml generate-costmap.py
	mkdir -p output
	python3 generate-costmap.py --input=$< --output=$@ --export_images --export_debug

install-requirements:
	pip3 install -r requirements.txt

test: output/berlin.npz read-costmap.py
	python3 read-costmap.py --input=$<

