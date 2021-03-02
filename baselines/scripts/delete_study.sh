#!/bin/bash
docker-compose -f docker/study-compose.yml down --remove-orphans
docker volume rm docker_mysql_volume