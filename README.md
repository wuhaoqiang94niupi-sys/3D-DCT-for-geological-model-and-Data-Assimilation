# 3D-DCT Geological Parameterization & History Matching

This repository contains the Python implementation of the 3D Discrete Cosine Transform (3D-DCT) framework for geological parameterization, as described in the manuscript:

> **"Advanced Spectrums decomposed for Geological Parameterization: High-Fidelity Characterization and Efficient Data Assimilation"**

## Overview

The code provides a high-fidelity, efficient method to parameterize complex 3D geological models (e.g., permeability fields) by transforming them into the frequency domain. This approach significantly reduces the dimensionality of the history matching problem while preserving geological continuity.

## Prerequisites

The scripts require the following Python libraries:

```bash
pip install numpy scipy matplotlib
