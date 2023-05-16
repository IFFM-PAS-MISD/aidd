#!/bin/bash

paper1.txt : a.txt pf_figures pk_figures
	@ echo "Producing paper 1"; touch paper1.txt;  echo "nested $(MYVAR)"

# contribution by pf
pf_figures: 
	@ echo "Running subsubmake contrib by pf"; $(MAKE) -f make_paper1_pf.mk

# contribution by pk
pk_figures: 
	@ echo "Running subsubmake contrib by pk"; $(MAKE) -f make_paper1_pk.mk
