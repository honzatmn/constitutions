#!/bin/bash


BASEDIR=$(dirname "$0")

cd "$BASEDIR"
cd webroot

python -m SimpleHTTPServer 8888
