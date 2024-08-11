#!/bin/bash
CONFIG_PATH=$1
# xserver permissions for docker
xhost +local:docker
# Setup named pipe for host process execution
if ! test -p $HOME/named_pipes/aci_execution_pipe; then
    mkdir $HOME/named_pipes
    mkfifo $HOME/named_pipes/aci_execution_pipe
fi
bash scripts/aci_execution_pipe.sh &> /dev/null &
# Run agent in docker container
export USER_ID="$(id -u)"
export GROUP_ID="$(id -g)"
export CONFIG_PATH=$CONFIG_PATH
docker compose --project-directory docker/ up --build
# Clean up processes on host machine
echo "shutdown_assetto_corsa" > $HOME/named_pipes/aci_execution_pipe
echo "shutdown_state_server" > $HOME/named_pipes/aci_execution_pipe
echo "shutdown_aci_execution_pipe" > $HOME/named_pipes/aci_execution_pipe