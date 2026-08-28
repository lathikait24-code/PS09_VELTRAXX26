# Problem Statement — PS-09

## INT8 Quantized CNN Deployment for Resource-Constrained Microcontrollers

### Problem Statement
Train and deploy a lightweight Convolutional Neural Network (CNN) for resource-constrained microcontrollers, using INT8 quantization to reduce memory and computational requirements.

### Requirements
- Train a lightweight CNN such as MobileNet or a custom ConvNet.
- Use CIFAR-10 or MNIST as the target dataset.
- Apply INT8 Post-Training Quantization (PTQ) to model weights.
- Generate a deployable model suitable for TinyML inference.
- Verify inference functionality using suitable test data.
- Compare the quantized model performance against the FP32 baseline.
- Demonstrate the model using an appropriate emulator or inference environment.

### Recommended Tools
- Python
- PyTorch / TensorFlow Lite Micro
- QEMU / Renode

### Expected Outcome
A functional INT8-quantized CNN model suitable for resource-constrained deployment, with verified inference results and performance comparison against the FP32 model.
# Final Report — PS-09: INT8 Quantized CNN Deployment

## 1. Objective
Train a lightweight CNN on CIFAR-10, quantize it to INT8 using post-training quantization, and compare it against the FP32 baseline for deployment on resource-constrained microcontrollers.

## 2. Pipeline Summary
1. **Dataset** — CIFAR-10, downloaded via `src/download_data.py`
2. **Training** — lightweight CNN trained and saved as `cifar_model.keras` *(training script pending — see Limitations)*
3. **Quantization** — `src/quantize_model.py` converts the trained model to:
   - `model_fp32.tflite` (FP32 baseline)
   - `model_int8.tflite` (INT8 PTQ, calibrated on 200 representative training samples)
4. **Verification** — `tb/test_pipeline.py` confirms both artifacts exist and reports size comparison

## 3. Results

| Metric | FP32 | INT8 |
|---|---|---|
| Model size | 56.26 KB | 0.53 KB |
| Compression ratio | — | **106.69x** |
| Accuracy | *(see `logs/results.txt`)* | *(see `logs/results.txt`)* |

## 4. Observations
- 106.69x is well above the typical ~4x compression expected from FP32→INT8 quantization (32-bit → 8-bit is a 4x weight-storage reduction). This suggests `cifar_model.keras` may be an unusually small/undertrained network, or a large share of its weights compress unusually well (e.g. near-zero or constant regions).
- Recommended follow-up: confirm `model.summary()` parameter count and check that FP32/INT8 accuracy in `logs/results.txt` are well above the 10% CIFAR-10 random baseline before treating the compression figure as representative.

## 5. Deployment Status
| Stage | Status |
|---|---|
| FP32 → INT8 PTQ | Complete |
| TFLite Micro (C array) export | Not yet implemented |
| QEMU/Renode emulated deployment | Not yet implemented |
| On-device inference verification | Not yet implemented |

## 6. Next Steps
- Export `model_int8.tflite` to a TFLite Micro C array (`xxd -i model_int8.tflite > model_data.cc`)
- Deploy to a QEMU or Renode target running the TFLite Micro runtime
- Run inference on sample test data on-device and record latency
- Update this report with on-device accuracy and latency figures
  <img width="3020" height="2070" alt="image" src="https://github.com/user-attachments/assets/ba263102-5b0c-49f2-88e4-9e7eadcaacb1" />
