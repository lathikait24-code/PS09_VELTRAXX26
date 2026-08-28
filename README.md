# INT8 Quantized CNN Deployment for Resource-Constrained Microcontrollers — VELTRAXX '26

## Project Overview

This project implements PS-09: training a lightweight CNN on CIFAR-10, applying INT8 post-training quantization (PTQ) to compress it for TinyML deployment, and verifying the quantized model's inference and performance against the FP32 baseline — targeting resource-constrained microcontrollers via a TFLite Micro / emulator (QEMU/Renode) deployment path.

## Repository Structure

## Requirements

- Python 3.9+
- TensorFlow (`tensorflow`) — model training, TFLite conversion, INT8 PTQ
- torchvision (`torchvision`) — used by `download_data.py` for the CIFAR-10 download

```bash
pip install tensorflow torchvision
```

## Instructions

### 1. Clone the repository
Use GitHub Desktop, or download as ZIP from the repo's **Code** button.

### 2. Download the dataset
Run `src/download_data.py` to fetch CIFAR-10 (and MNIST).

### 3. Train the base CNN
Train a lightweight CNN (custom ConvNet) on CIFAR-10 and save it as `cifar_model.keras` in the project root — `quantize_model.py` expects this file to exist before running. *(Training script not yet included in this repo.)*

### 4. Quantize and evaluate
Run `src/quantize_model.py`. This:
- Evaluates the trained model's FP32 accuracy
- Converts it to `model_fp32.tflite` and `model_int8.tflite` (INT8 PTQ, calibrated on 200 training samples)
- Evaluates INT8 accuracy on the test set
- Writes `logs/results.txt` with accuracy, size, and compression numbers

### 5. Verify the pipeline output
Run `tb/test_pipeline.py` to confirm both `.tflite` files exist and print the size comparison.

**Latest verified output:**
