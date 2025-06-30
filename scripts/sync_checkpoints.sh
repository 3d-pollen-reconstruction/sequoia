#!/bin/bash

# Define remote user and host
REMOTE_USER="n.fahrni"
REMOTE_HOST="slurmlogin.cs.technik.fhnw.ch"
REMOTE_PATH="/home2/n.fahrni/sequoia/checkpoints"

# Define local destination path
LOCAL_DESTINATION="."

# Use scp to copy the folder recursively
scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" "${LOCAL_DESTINATION}"


# scp -r n.fahrni@slurmlogin.cs.technik.fhnw.ch:/home2/n.fahrni/sequoia/checkpoints ./sequoia
# rsync -avz --progress n.fahrni@slurmlogin.cs.technik.fhnw.ch:/home2/n.fahrni/sequoia/checkpoints/ ./sequoia/checkpoints/
