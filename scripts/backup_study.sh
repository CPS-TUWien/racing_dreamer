#!/bin/bash
database=$1
file=$2
docker exec docker_database_1 /usr/bin/mysqldump -u root --password=password $database > $file
