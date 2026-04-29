<div align="center">

# 🚀 OGSA-YOLO: Omni-directional Gated Spatial-Attention Framework

**End-to-end instance segmentation and information fusion framework for conveyor belt monitoring in complex underground coal mines.**
<br>
Submitted to _The Visual Computer_ (TVC)

[![DOI](https://zenodo.org/badge/DOI/YOUR_DOI_HERE.svg)](https://doi.org/YOUR_DOI_HERE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## 🌟 Introduction

The precise parsing of the conveyor belt's operating status constitutes the core challenge in intelligent mining. Existing detection technologies still face severe challenges when confronting complex scenarios typical of underground coal mines, such as **high dust concentration, extremely low illumination, and motion blur**.

To address these issues, we propose **OGSA-YOLO**, an end-to-end instance segmentation and information fusion framework based on YOLOv11. It fundamentally resolves multi-scale feature perception conflicts and efficiently suppresses unstructured noise under complex working conditions.

## ✨ Core Innovations

Our proposed framework integrates four novel core components, which are implemented in `CDA.py`:

- **ODP-Conv (Omni-Directional Perception Dynamic Convolution)**: Introduced in the shallow and neck networks to endow the model with explicit geometric direction perception capability via asymmetric dynamic weights.
- **GSDF-Block (Gated Spatio-Dynamic Fusion Block)**: Utilized in the deep layers to intelligently aggregate global context through an adaptive gating mechanism, ensuring the feature integrity of large-scale targets.
- **SCSAB (Shared Context-Aware Self-Attention Block)**: Utilizes orthogonal strip convolution priors and adaptive hybrid channel attention to tackle severe environmental interference and preserve fragile target edge information.
- **AMFAM (Adaptive Multi-scale Fusion and Attention Module)**: Introduces category semantics as guidance signals to dynamically coordinate the on-demand fusion of multi-scale features, thoroughly alleviating semantic misalignment.

---

## 🛠️ Installation

**1. Prepare the environment**
We recommend using Python 3.9+ and PyTorch 2.0+.

```bash
# Clone this repository
git clone [https://github.com/YOUR_GITHUB_USERNAME/OGSA-YOLO.git](https://github.com/YOUR_GITHUB_USERNAME/OGSA-YOLO.git)
cd OGSA-YOLO

# Create conda environment
conda create -n ogsayolo python=3.9 -y
conda activate ogsayolo

# Install dependencies
pip install -r requirements.txt
```
