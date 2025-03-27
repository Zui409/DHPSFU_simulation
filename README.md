# DHPSFU_simulation
Scripts to generate simulation DHPSF data for comparing DHPSFU with other algorithms.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Software
------------
ImageJ

Matlab R2024B

Spyder (python 3.12)

ImageJ Plug-in: GDSC SMLM

Description
------------
These scripts are developed for generating simulated double-helix PSF (DH-PSF) datasets, designed to benchmark the performance of DHPSFU against other algorithms.

Workflow Overview
------------
The full simulation and evaluation workflow includes the following stages:

Simulation: Simulated DH-PSFs are generated using a retrieved phase mask derived from real microscope images. These DH-PSFs can be placed at arbitrary x,y,z coordinates relative to the origin. The simulated PSFs are randomly positioned on a black canvas with user-defined noise levels. The exact 3D positions (ground truth) of each PSF are recorded for later evaluation.

Localisation Extraction: Simulated datasets are processed using the PeakFit function from the GDSC SMLM suite and the DHPSFU algorithm to extract localisation coordinates.

Performance Comparison: Extracted results are compared against the ground truth to calculate key performance metrics, including localisation precision, detection accuracy, and sensitivity.

Instruction
------------
Install Required Packages:Ensure all required dependencies (MATLAB, Python, Fiji/ImageJ with GDSC SMLM, and optionally ThunderSTORM) are properly installed.

Step 1: Simulate DH-PSFs (MATLAB)
Run the MATLAB script to generate DH-PSFs from a retrieved phase mask. This step will produce:#
- Calib.TIFF: A simulated calibration stack.
- Beads_with_shift.TIFF: Simulated DH-PSFs with random sub-pixel XY shifts.
- GT_shift.csv: The ground truth file listing the x,y shifts and z positions for each frame.

Step 2: Generate Synthetic Datasets (Python)
Use the provided Python script to simulate realistic experimental datasets. Customise parameters such as: number of DH-PSFs per frame, total number of frames, canvas size, background noise levels intensity fluctuation range etc.
This allows for flexible dataset generation tailored to experimental conditions.

Step 3: Ground Truth Generation (Python)
Create a corresponding ground truth file for each simulated dataset to be used in performance evaluation.

Step 4: Localisation with GDSC SMLM (Fiji/ImageJ)
Use the PeakFit plugin in GDSC SMLM to extract raw 2D localisations from both the calibration and dataset TIFF stacks.
Optional: ThunderSTORM Users
If localisations are extracted using ThunderSTORM, run Opt_ThunderSTORM_to_GDSC_PF_convert.py to convert the results into GDSC PeakFit format compatible with DHPSFU.

Step 5: 3D Localisation with DHPSFU (Python)
Run the DHPSFU algorithm using the peak-fitted localisations and the simulated calibration file to obtain accurate 3D coordinates.

Step 6: Compare with Ground Truth (Python)
Use Compare_GT.py to evaluate the results by comparing DHPSFU output with the ground truth. The script will compute and report localisation precision, detection accuracy, and sensitivity.
