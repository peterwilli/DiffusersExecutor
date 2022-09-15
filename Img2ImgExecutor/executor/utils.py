import gc
import torch
import os
import pathlib

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()