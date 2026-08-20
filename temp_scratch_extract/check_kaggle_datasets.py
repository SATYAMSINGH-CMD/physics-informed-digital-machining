import subprocess
import os

print("Testing Kaggle downloads for Datasets 1 to 25:")
for ds in range(1, 26):
    cmd_h5 = f'kaggle datasets download -d tonylschmitz/digital-machining-database -f "Dataset {ds} h5/stability_boundary{ds}.h5" -p temp_scratch_extract'
    res_h5 = subprocess.run(cmd_h5, shell=True, capture_output=True, text=True)
    
    cmd_mat = f'kaggle datasets download -d tonylschmitz/digital-machining-database -f "Dataset {ds} mat/stability_boundary{ds}.mat" -p temp_scratch_extract'
    res_mat = subprocess.run(cmd_mat, shell=True, capture_output=True, text=True)
    
    status = []
    if res_h5.returncode == 0:
        status.append("h5")
    if res_mat.returncode == 0:
        status.append("mat")
        
    res_str = ", ".join(status) if status else "NOT FOUND / 404 on Kaggle"
    print(f"Dataset {ds:2d}: {res_str}")
