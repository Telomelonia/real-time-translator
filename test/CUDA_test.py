import torch
print("CUDA available:", torch.cuda.is_available())
print("PyTorch Version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")