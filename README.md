# Reinforcing the Weakest Links: Modernizing SIENA

This repository contains the code, modified pipelines, and scripts associated with the paper **"Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration"**.

The traditional SIENA (Structural Image Evaluation, using Normalisation, of Atrophy) pipeline is widely used to estimate the Percentage Brain Volume Change ($\text{PBVC}$) between longitudinal MRI scans. This project modernizes the pipeline by replacing its most error-prone classical components with robust Deep Learning (DL) alternatives: **SynthStrip** for skull stripping and **SynthSeg** for tissue segmentation.

## Pipeline Variants

The repository provides four distinct versions of the SIENA pipeline to evaluate the impact of each DL integration:

* **`SIENA_VANILLA`**: The standard, unmodified FSL SIENA pipeline using BET for skull stripping and FAST for segmentation.
* **`SIENA-SS`**: Integrates DL-based skull stripping (SynthStrip) while retaining FAST for segmentation.
* **`SIENA-SEG`**: Integrates DL-based tissue segmentation (SynthSeg) while retaining BET for skull stripping.
* **`SEINA-SS-SEG`** *(SIENA-SS-SEG)*: The fully modernized pipeline utilizing both SynthStrip and SynthSeg for maximum robustness.

## Repository Structure

### Core Python Modules

* **`findskull.py`**: A custom Python script that processes brain masks to extract surface geometry, trace intensity probes, and reconstruct the skull volume using operations like binary erosion and Gaussian filtering.
* **`synthseg.py`**: A Python wrapper that executes FreeSurfer's `mri_synthseg`. It processes the resulting segmentations to extract specific tissue probability maps (Gray Matter, White Matter, and CSF) for the pipeline.

### Modified SIENA Components

* **`siena_diff_x_siena_diff.cc`**: The modified C++ source code responsible for computing brain change using edge motion and segmentation.
* **`siena_diff`**: The compiled binary of the aforementioned C++ code.
* **`siena_cal_sienadiff`**: A modified shell script handling the self-calibration step of the SIENA differential analysis.

## Prerequisites

To run these modernized pipelines, the following dependencies are required:

* **FSL** (FMRIB Software Library)
* **FreeSurfer** (specifically the modules containing `mri_synthseg` and `mri_synthstrip`)
* **Python 3.x** with the following packages:
* `numpy`
* `scipy`
* `nibabel`



## Usage

Ensure all dependencies are correctly loaded in your environment path. You can execute any of the pipeline variants identically to the standard FSL `siena` command. For example, to run the fully modernized pipeline on a baseline scan ($I_1$) and a follow-up scan ($I_2$):

```bash
./SEINA-SS-SEG <scan1.nii.gz> <scan2.nii.gz> [options]

```
