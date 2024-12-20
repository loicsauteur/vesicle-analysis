# vesicle-analysis

[![License](https://img.shields.io/pypi/l/vesicle-analysis.svg?color=green)](https://github.com/loicsauteur/vesicle-analysis/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/vesicle-analysis.svg?color=green)](https://pypi.org/project/vesicle-analysis)
[![Python Version](https://img.shields.io/pypi/pyversions/vesicle-analysis.svg?color=green)](https://python.org)
[![CI](https://github.com/loicsauteur/vesicle-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/loicsauteur/vesicle-analysis/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/loicsauteur/vesicle-analysis/branch/main/graph/badge.svg)](https://codecov.io/gh/loicsauteur/vesicle-analysis)

A custom package for vesicle analysis.

# Description

Custom analysis on vesicles. Requires small images of individual cells, with nucleus and vesicle marker.

Run the analysis with the [workflow notebook](src/vesicle_analysis/notebooks/workflow.ipynb).

# Installation

1. Create an environment (only python 3.9 and 3.10 supported), e.g. with conda:

`conda create -n vesicle-analysis python=3.9`

`conda activate vesicle-analysis`

2. Install the package
  
    2. a) Install directly from Git
    - with: `pip install git+https://github.com/loicsauteur/vesicle-analysis`
    - upgrade to the latest version:
    `pip install --upgrade git+https://github.com/loicsauteur/vesicle-analysis`
   
    2 b) Download and install locally
    - cd to the downloaded code folder
    - Install the package with:
    `pip install -e .`
    
    2 c) Install Dev/Test dependencies
    - download the code and type:
    `pip install -e ".[test,dev]"`



<!---
Note
**check possibility to directly install from github (via pip)**
-->