#!/bin/bash
# Make is a tool developed in 70! But it has easy syntax and is still popular. 
# When it is time to execute commands to update a target, they are executed by making a new subshell for each line
# ****************************************************
# This is an exemplary Makefile for a reaserch project
# 		written by 
# 		Pawel Kudela
# ****************************************************
# The main aim is to link figures in publications with data, automate workflow tracking all dependencies
# and build pdf for target papers
# 'make' build main_2023_mssp_paper.pdf
# 'make doc' build main_2023_mssp_paper.pdf
# 'make clean' removes latex build files (latexmk_aux)
# The @ at the beginning of the receipe means that the command itself isn't displayed in the terminal

# ***** Variables - environments *****
CONDA_ENV := cookie_env
# ***** Variables - paths *****
CONDA_ENV_PATH := /usr/local/anaconda3/
VIZ := ./src/visualization/
PROC := ./src/data_processing/
TOOLS := ./src/tools/
FIG := ./reports/figures/
PAPERS := ./reports/journal_papers/
# ***** Variables - executables *****
MATLAB := /usr/local/MATLAB/R2023a/bin/matlab -nodesktop -nosplash -nodisplay
# ***** Variables - objects *****
# DISPOSABLE_OBJS = ...
# PRECIOUS_OBJS = ...
# OBJS = $(DISPOSABLE_OBJS) $(PRECIOUS_OBJS)
# Stop GNU make from overzealous deletion of intermediate files
.PRECIOUS: %.dvi %.ps %.pdf %.png
# ****************************************************
help : doc
	@echo "----------------HELP----------------"
	@echo "To build project type: make"
	@echo "To build pdf files type: make doc"
	@echo "To clean latex aux files type: make clean"
# ****************************************************	
# 		2023_mssp_paper
# ****************************************************
PAPERFOLDER = $(PAPERS)2023_mssp_paper/
# doc is pseudo-target (phony target); It is not really the name of a file; 
# rather it is just a name for a recipe to be executed
doc : $(PAPERFOLDER)main_2023_mssp_paper.pdf

# compile latex document
$(PAPERFOLDER)main_2023_mssp_paper.pdf : $(PAPERFOLDER)main_2023_mssp_paper.tex $(PAPERFOLDER)2023_mssp_paper_abstract.tex \
$(PAPERFOLDER)2023_mssp_paper_introduction.tex $(PAPERFOLDER)2023_mssp_paper_methodology.tex $(PAPERFOLDER)2023_mssp_paper_results.tex \
$(PAPERFOLDER)2023_mssp_paper_conclusions.tex $(PAPERFOLDER)2023_mssp_paper_bibliography.bib \
$(PAPERFOLDER)figs/figure1.png $(PAPERFOLDER)figs/figure2.png
	cd $(PAPERFOLDER) && latexmk -pdf -bibtex -auxdir=latexmk_aux -pv main_2023_mssp_paper.tex # -pv for preview
#	(cd $(PAPERFOLDER); latexmk -pdf -bibtex -auxdir=latexmk_aux main_2023_mssp_paper.tex) # without pdf preview

# copy and rename figures
$(PAPERFOLDER)figs/figure2.png : $(FIG)plot_rms/plot_rms_specimen_1_50kHz_pzt_rms_norm.png	
	cd $(TOOLS) && ./copy_figs_2023_mssp_paper.sh
	
$(PAPERFOLDER)figs/figure1.png : $(FIG)plot_rms/plot_rms_specimen_1_50kHz_pzt_rms.png	
	 cd $(TOOLS) && ./copy_figs_2023_mssp_paper.sh

# data visualization in MATLAB
$(FIG)plot_rms/plot_rms_specimen_1_50kHz_pzt_rms.png $(FIG)plot_rms/plot_rms_specimen_1_50kHz_pzt_rms_norm.png : $(VIZ)plot_rms.m \
./data/processed/specimen_1/rms/50kHz_pzt_rms.mat ./data/processed/specimen_1/rms_norm/50kHz_pzt_rms_norm.mat
	@ $(MATLAB) -r "try, run('$(VIZ)plot_rms'); catch, warning('Problem running Matlab script: plot_rms'); end, exit" \
	&& sleep 3 # writing images to hard drive may take a while so we are waiting 3s
	
# data processing in Python
./data/processed/specimen_1/rms_norm/50kHz_pzt_rms_norm.mat : $(PROC)rms_norm.py ./data/raw/specimen_1/50kHz_pzt.mat
	cd $(PROC) && conda run -n $(CONDA_ENV) python3 rms_norm.py

# data processing in MATLAB	
# Build target only if it does not exist despite that the time stamp of dependencies has changed; 
# useful if computations to obtain the target are long

ifeq (,$(wildcard ./data/processed/specimen_1/rms/50kHz_pzt_rms.mat)) 
FILE_EXISTS = false ; $(info File 50kHz_pzt_rms.mat does not exist - recomputing)
./data/processed/specimen_1/rms/50kHz_pzt_rms.mat : $(PROC)rms.m ./data/raw/specimen_1/50kHz_pzt.mat
	$(MATLAB) -r "try, run('$(PROC)rms'); catch, warning('Problem running Matlab script: rms'); end, exit" 
else
./data/processed/specimen_1/rms/50kHz_pzt_rms.mat : $(PROC)rms.m ./data/raw/specimen_1/50kHz_pzt.mat
	@ echo rms.m or 50kHz_pzt.mat has changed but file 50kHz_pzt_rms.mat exist - do nothing
endif 
# ****************************************************
.PHONY: clean help doc
clean :
	rm -f -r $(PAPERFOLDER)latexmk_aux
#	-rm -f $(DISPOSABLE_OBJS) # dash at the beginning means ignore errors
#	rm -f $(filter-out keep1.o keep2.o,$(OBJS)) # to remove all $(OBJS) except keep1.o and keep2.o
