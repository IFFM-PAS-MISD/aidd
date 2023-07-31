#!/bin/bash

# contribution by pf
pf_figures: fig1 fig2

fig2: 
	@ echo "Producing fig2"; touch fig2
	
fig1: 
	@ echo "Producing fig1"; touch fig1; echo "Double nested: $(MYVAR)"

