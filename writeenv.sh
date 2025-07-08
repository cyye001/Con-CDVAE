#!/bin/bash

CURRENT_PATH=$(pwd)

sed -i "s|YOUR_PATH_TO_CONCDVAE|$CURRENT_PATH|g" .env

echo "update the .env file：$CURRENT_PATH"
