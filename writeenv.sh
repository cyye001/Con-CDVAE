#!/bin/bash

# Environment setup script for Con-CDVAE
# This script updates the .env file with the current working directory path
# It replaces the placeholder 'YOUR_PATH_TO_CONCDVAE' with the actual project path

# Get the current working directory
CURRENT_PATH=$(pwd)

# Replace the placeholder in .env file with the actual path
sed -i "s|YOUR_PATH_TO_CONCDVAE|$CURRENT_PATH|g" .env

# Print confirmation message
echo "update the .env file：$CURRENT_PATH"
