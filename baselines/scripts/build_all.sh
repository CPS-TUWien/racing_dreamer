#!/bin/bash
docker build -t tuwcps/racing:acme -f docker/Dockerfile.acme .
docker build -t tuwcps/racing:sb3 -f docker/Dockerfile.sb3 .
docker build -t tuwcps/racing:sb2 -f docker/Dockerfile.sb2 .
