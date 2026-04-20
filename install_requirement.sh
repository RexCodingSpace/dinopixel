#!/bin/bash
set -e

# 1. PyTorch (Jetson wheel, 從 NVIDIA 下載)
pip install https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl

# 2. 其他所有套件(含 torchvision 和 onnxruntime-gpu,從 Jetson AI Lab)
pip install -r requirements.txt --extra-index-url https://pypi.jetson-ai-lab.dev/jp6/cu122

# 3. 驗證
python -c "
import torch, tensorrt, onnxruntime as ort
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('TensorRT:', tensorrt.__version__)
print('ORT providers:', ort.get_available_providers())
"