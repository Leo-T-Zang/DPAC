import os
import torch
import random
import numpy as np
from torch import nn
import pandas as pd

class ModelConfig:
    def __init__(self):

        self.na_embd = 1024 # NucleotideTransformer
        self.prot_embd = 1280 #ESM2
        self.bias = True,
        self.init_logit_bias = None
        self.init_t = 0.07


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
