import gc
import torch
import os
import pathlib
import gdown

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

def download_get_sd_model():
    script_path = pathlib.Path(__file__).parent.resolve()
    sd_path = os.path.join(script_path, "sd")
    if not os.path.exists(sd_path):
        print("Need to download SD!")
        os.chdir(script_path)
        download_link = "https://drive.google.com/uc?export=download&id=1M4-Cgl7PmpGdyhebpL6b18oYBDyjggVA"
        # os.system(f"wget {download_link} -O sd.tar.gz")
        gdown.download(download_link, "sd.tar.gz", quiet=False)
        pathlib.Path(sd_path).mkdir(parents=True, exist_ok=True)
        os.system(f"tar -zxf sd.tar.gz -C {sd_path}")
    return sd_path