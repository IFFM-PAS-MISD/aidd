#!/bin/bash
echo "Copying and renaming files"
cd ../.. # go up two levels in the directory tree
# cp ./reports/figures/plot_rms/plot_rms_specimen_1_50kHz_pzt_rms.png ./reports/journal_papers/2023_mssp_paper/figs/figure1.png # copy
# cp ./reports/figures/plot_rms/plot_rms_specimen_1_50kHz_pzt_rms_norm.png ./reports/journal_papers/2023_mssp_paper/figs/figure2.png # copy
mkdir -p ./reports/journal_papers/2023_mssp_paper/figs && ln -sf $(pwd)/reports/figures/plot_rms/plot_rms_specimen_1_50kHz_pzt_rms.png $(pwd)/reports/journal_papers/2023_mssp_paper/figs/figure1.png # symbolic link
mkdir -p ./reports/journal_papers/2023_mssp_paper/figs && ln -sf $(pwd)/reports/figures/plot_rms/plot_rms_specimen_1_50kHz_pzt_rms_norm.png $(pwd)/reports/journal_papers/2023_mssp_paper/figs/figure2.png # symbolic link
