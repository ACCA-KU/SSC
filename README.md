# Solvatochromic Subgroup Contribution (SSC) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a graph neural network for molecular property prediction, reflecting the idea of "Solvatochromic subgroup contribution". In the approach of SSC, the solvent effects on the molecular property can be quantified by the contirbution of consisting functional groups. 

This project is built upon our previous project [D4CMPP](https://github.com/ACCA-KU/DeepMPP), which is the deep learning pipeline built with PyTorch and Deep Graph Library (DGL)

## Installation
```bash
pip install git+https://github.com/ACCA-KU/SSC.git
```
or
```bash
git clone https://github.com/ACCA-KU/SSC.git
cd SSC
pip install -e
```
## Quick Start
```python
from SSC import train
train(data="CSV_file_name", target=["Column_name",])
```
"example.ipynb" provides the example codes for training and prediction of sample dataset.

This single command automatically supports the all steps of trainning model, including preprocessing, trainning, and logging the result.

Please refer the repository of [D4CMPP](https://github.com/ACCA-KU/DeepMPP) for more details.
