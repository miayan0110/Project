import torch
import subprocess
import re

def list_total_gpu_usage():
    if not torch.cuda.is_available():
        print("No GPU detected.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Total GPUs: {num_gpus}\n")
    
    # 使用 nvidia-smi 查詢所有 GPU 記憶體使用狀況
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, text=True
        )
        smi_output = result.stdout.strip().split("\n")
        
        for i, line in enumerate(smi_output):
            device_info = line.split(", ")
            device_name = device_info[0]
            total_memory = int(device_info[1])  # MB
            used_memory = int(device_info[2])   # MB
            free_memory = int(device_info[3])   # MB

            # torch.cuda.set_device(i)  # 確保 PyTorch 查詢正確的 GPU
            # pytorch_allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
            # pytorch_reserved = torch.cuda.memory_reserved(i) / 1024**2  # MB
            
            print(f"GPU {i}: {device_name}")
            print(f"  Total Memory: {total_memory} MB")
            print(f"  Used Memory (Total): {used_memory} MB")
            print(f"  Free Memory: {free_memory} MB\n")
            # print(f"  PyTorch Allocated: {pytorch_allocated:.2f} MB")
            # print(f"  PyTorch Reserved: {pytorch_reserved:.2f} MB\n")

    except FileNotFoundError:
        print("nvidia-smi command not found. Make sure you have NVIDIA drivers installed.")

# 呼叫函數查看完整 GPU 使用狀況
list_total_gpu_usage()
