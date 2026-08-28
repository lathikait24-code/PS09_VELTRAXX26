# src/ — Source Code

Model training, quantization, and data pipeline.

| File | Purpose |
|---|---|
| `download_data.py` | Fetch CIFAR-10 / MNIST via torchvision |
| `train_model.py` | Train lightweight CNN → `cifar_model.keras` |
| `quantize_model.py` | FP32 eval, TFLite + INT8 PTQ conversion, INT8 eval |

Run in order: download → train → quantize.
For RTL variants, place `.v` / `.sv` / `.vhd` sources here
