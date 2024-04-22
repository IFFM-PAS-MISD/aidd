#!/bin/bash
SOURCE_FILE=$(ls Surrogate_Modeling_Paper.tex)
DESTINATION_FILE="$HOME/grammarly_web/detex-out.txt"
detex -l $SOURCE_FILE > $DESTINATION_FILE
