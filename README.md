# YOLOv7 Setup Guide

This guide will walk you through setting up the environment required to run YOLOv7 on a machine with CUDA support.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [CUDA Installation](#cuda-installation)
- [YOLOv7 Installation](#yolov7-installation)
- [Verification](#verification)

## Prerequisites

Before you start, ensure that you have the following:

- A machine with a CUDA-capable GPU.
- `python3.8+` installed.
- Basic knowledge of Python and command-line interface (CLI) operations.

## Environment Setup

1. **Clone the YOLOv7 Repository**

   ```bash
   
    git clone https://github.com/WongKinYiu/yolov7.git
2. **Create a Python Virtual Environment**

It’s recommended to use a virtual environment to avoid conflicts with other projects.
    ```bash
    
    python3 -m venv yolov7-env
    source yolov7-env/bin/activate  # On Windows use yolov7-env\Scripts\activate

3. **Upgrade pip**
    ```bash

    pip install upgrade pip

# CUDA Installation
If you don't have CUDA installed, follow these steps:

1. **Check for CUDA Compatibility**

Make sure your GPU is compatible with CUDA by checking [NVIDIA's CUDA GPUs list](https://developer.nvidia.com/cuda-gpus).

2. **Install CUDA Toolkit**

Download and install the appropriate CUDA version from the [NVIDIA CUDA Toolkit.](https://developer.nvidia.com/cuda-toolkit)

Follow the installation instructions provided by NVIDIA for your operating system.

3. **Install cuDNN**

Download cuDNN from the [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) page and follow the installation guide provided for your CUDA version.

4. **Verify CUDA Installation**

After installation, verify that CUDA is properly installed:
    ```bash
    
    nvcc --version


    












