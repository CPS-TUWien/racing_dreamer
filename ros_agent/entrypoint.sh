#!/bin/bash
set -e

echo "-------------------- entrypoint.sh debug info follows. --------------------"

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
test -f /ws/devel/setup.bash && source /ws/devel/setup.bash

if [ ! -z "${AGENT_NAME}" ]
then
	echo "AGENT_NAME = ${AGENT_NAME}"
	cp -av /cwd/agents/${AGENT_NAME}/src/agent.py /ws/src/f1tenth_agent_ros/src/agent.py
else
	echo "AGENT_NAME not set"
fi

if [ ! -z "${ROS_NAMESPACE}" ]
then
	EXTRA_ARGS="${EXTRA_ARGS} --namespace=${ROS_NAMESPACE}"
fi

if [ ! -z "${AGENT_SELECT}" ]
then
	EXTRA_ARGS="${EXTRA_ARGS} --agent=${AGENT_SELECT}"
fi

if [ ! -z "${AGENT_CHECKPOINT}" ]
then
	EXTRA_ARGS="${EXTRA_ARGS} --checkpoint=/cwd/checkpoints/${AGENT_CHECKPOINT}"
fi

if uname -a | grep -q "tegra "
then
	EXTRA_ARGS="${EXTRA_ARGS} --hardware=car"
else
	EXTRA_ARGS="${EXTRA_ARGS} --hardware=other"
fi

echo "EXTRA_ARGS = ${EXTRA_ARGS}"
echo "@ = $@"

echo "-------------------- entrypoint.sh done, executing command. --------------------"

#export ROS_IP=192.168.1.150
#export ROS_MASTER_URI=http://192.168.1.102:11311

if [[ "/bin/bash" == "${@}" ]]
then
	exec "${@}"
elif [ ! -z "${EXTRA_ARGS}" ] 
then
	exec "${@} ${EXTRA_ARGS}"
else	
	exec "${@}"
fi
