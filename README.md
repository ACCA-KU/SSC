# Solvatochromic Group Contribution (SGC) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a graph neural network for molecular property prediction, reflecting the idea of "Solvatochromic group contribution". In the approach of SGC, the solvent effects on the molecular property can be quantified by the contirbution of consisting functional groups. 

This project is built upon our previous project [D4CMPP](https://github.com/spark8ku/DeepMPP), which is  the deep learning pipeline built with PyTorch and Deep Graph Library (DGL)

## Installation
```bash
pip install git+https://github.com/phillip1998/SGC.git
```
or
```bash
git clone https://github.com/phillip1998/SGC.git
cd SGC
pip install -e
```
## Quick Start
```python
from SGC import train
train(data="CSV_file_name", target=["Column_name",])
```
"example.ipynb" provides the example codes for training and prediction of sample dataset.

This single command automatically supports the all steps of trainning model, including preprocessing, trainning, and logging the result.

Please refer the repository of [D4CMPP](https://github.com/spark8ku/DeepMPP) for more details.
