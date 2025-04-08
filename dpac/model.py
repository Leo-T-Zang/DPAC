import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

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

seed = 42
# set_seed(seed)

#########################################################
# Model
##########################################################

class MLP(nn.Module):
    def __init__(self, emb_dim_in, emb_dim_out, bias):
        super().__init__()
        self.c_fc = nn.Linear(emb_dim_in, 4 * emb_dim_out, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * emb_dim_out, emb_dim_out, bias=bias)

    def forward(self, prot):
        prot = self.c_fc(prot)
        prot = self.gelu(prot)
        prot = self.c_proj(prot)
        return prot

# Define the main model
class DPAC(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.na_mlp = MLP(config.na_embd, config.prot_embd, config.bias)
        self.prot_mlp = MLP(config.prot_embd, config.prot_embd, config.bias)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / config.init_t))

        if config.init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * config.init_logit_bias)
        else:
            self.logit_bias = None
        
        self.ln_final = nn.LayerNorm(config.prot_embd)
        #self.initialize_parameters()

    def initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def encode_na(self, na, normalize: bool = False):
        features = self.na_mlp(na)
        features = self.ln_final(features)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_prot(self, prot, normalize: bool = False):
        features = self.prot_mlp(prot)
        features = self.ln_final(features)
        return F.normalize(features, dim=-1) if normalize else features

    def get_logits(self, prot, na):
        na_features = self.encode_na(na, normalize=True)
        prot_features = self.encode_prot(prot, normalize=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_prot = logit_scale * prot_features @ na_features.t()
        if self.logit_bias is not None:
            logits_per_prot += self.logit_bias
        logits_per_na = logits_per_prot.t()

        return logits_per_prot, logits_per_na

    def forward(self, prot, na):
        logits_per_prot, logits_per_na = self.get_logits(prot, na)
        return logits_per_prot, logits_per_na

