import os

cuda_paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
    r"C:\Windows\System32",
    os.environ.get('PATH', '').split(';')
]

dll_files = [
    'cudnn64_8.dll',
    'cudnn_ops_infer64_8.dll',
    'cudnn_ops_train64_8.dll',
    'cudnn_cnn_infer64_8.dll'
]

for path in cuda_paths:
    if path:
        print(f"\nChecking path: {path}")
        if os.path.exists(path):
            files = os.listdir(path)
            for dll in dll_files:
                if dll in files:
                    print(f"Found {dll}")
                else:
                    print(f"Missing {dll}")
        else:
            print("Path does not exist")