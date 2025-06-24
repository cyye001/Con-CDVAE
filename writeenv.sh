#!/bin/bash

CURRENT_PATH=$(pwd)

sed -i "s|YOUR_PATH|$CURRENT_PATH|g" .env

echo "update the .env file：$CURRENT_PATH"
