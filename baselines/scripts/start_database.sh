#!/bin/bash
export STUDY=$1
data=$2
docker-compose -f docker/study-compose.yml up -d database
sleep 10
docker exec -i docker_database_1 /usr/bin/mysql -u user --password=password $1 < $2