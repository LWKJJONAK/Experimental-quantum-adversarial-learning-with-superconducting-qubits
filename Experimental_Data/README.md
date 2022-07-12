# This folder stores the main data collected during the experiments as well as the corresponding codes for the data analysis.

Required packages: numpy, matplotlib

Steps to plot the figures (skip step 1 if pylabrad is already installed):
1. Install labrad:

    pip install pylabrad

2. Run script:

    python run_plot.py

3. Select from the following items to plot:
    - MEDICAL: experiment results for learning MRI MNIST data of hands and breasts, i.e., figures in Fig.2 and Fig.4 in the main text.
    - QUANTUM: experiment results for learning quantum data, i.e., figures in Fig. 3 in the main text.
    - MNIST: experiment results for learning digital MNIST data, i.e., figures in Fig. S13 in the supplementary material.
