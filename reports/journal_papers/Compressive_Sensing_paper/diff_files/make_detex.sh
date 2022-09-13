#!/bin/bash
SOURCE_FILE=$(ls main*.tex)
DESTINATION_FILE="$HOME/grammarly_web/detex-out.txt"
detex -l $SOURCE_FILE > $DESTINATION_FILE
