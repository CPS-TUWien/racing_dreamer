#!/bin/bash
docker exec $1 /usr/bin/mysqldump -u root --password=password mpo-study > $1.sql
