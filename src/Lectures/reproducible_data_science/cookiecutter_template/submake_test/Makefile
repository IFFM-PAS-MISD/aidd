#!/bin/bash

ifndef VERBOSE
MAKEFLAGS += --no-print-directory
endif

export MYVAR := /subfolder

all: paper1 paper2

paper1:	
	@ echo "Running submake 1"; $(MAKE) -f make_paper1.mk
paper2:
	@ echo "Running submake 2"; $(MAKE) -f make_paper2.mk 

.PHONY: paper1 paper2
