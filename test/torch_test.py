import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Test CUDA tensor operations
if torch.cuda.is_available():
    x = torch.tensor([1., 2.]).cuda()
    print("CUDA Tensor test:", x)