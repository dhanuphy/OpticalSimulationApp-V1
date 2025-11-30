# Optical Simulator App

Overview
This software provides a complete workflow for analyzing optical transmittance spectra of thin films. It includes manual and automatic envelope extraction, rough and fine thickness estimation, optical simulation, and extraction of more than 20 optical parameters.

Key Features
1) Manual Envelope Selection (Tmax/Tmin), - Add and edit envelope points, - Cubic/linear interpolation, - Automated Peak Detection, - Auto-selects Tmax and Tmin from raw spectra, - Rough Thickness Estimation, - Based on Swanepoel formula using only wavelengths, - Fine Automatic Simulation, - Full-curve RMSE optimization, - User-Friendly GUI, - PyQt5+ Matplotlib with real-time cursor readout.

Required installation
1. Install Python 3.8+. 2. Install packages: pip install pyqt5, numpy, scipy, matplotlib, pandas

Exported Output
Generates Optical_Properties.csv containing: - Experimental data, - Envelope data, - Simulated spectra, - Optical constants and bandgap-related values. - Optical Properties Exported - n, k, e1, e2, OD, sigma, skin depth, alpha, alphaE, (alphaE)^0.5, (alphaE)^2, derivative-alpha, log-alpha.

Workflow: 
Step 1 — Load Data. Step 2 — Manual envelope selection, select either Tmax or Tmin, can drag and adjust points to generate good envelope. Step 3 — Auto peak detection. Step 4 — Rough Simulation. Step 5 — Fine Auto Fit, press multiple times to generate fine spectra.  Step 6 — Export Optical Properties.
Credits

Developed by: Dhanunjaya Munthala, Institute of Research and Development (IRD), Suranaree University of Technology (SUT), Thailand.
Assistance: ChatGPT

License
This software may be released under the MIT License or license of your choice.


### 1. Clone the repository
