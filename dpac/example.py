import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import esm
import os
from .model import DPAC
from .utils import ModelConfig


# Example of using DPAC for scoring

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config = ModelConfig()
scoring_model = DPAC(model_config).to(device)
dpac_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
weight_path = os.path.join(dpac_dir, 'dpac/weight.pt')
scoring_model.load_state_dict(torch.load(weight_path)['model_state_dict'])
scoring_model.eval()

dna_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)
dna_model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True).to(device)
dna_model.eval()

protein_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
protein_model = protein_model.to(device)
protein_model.eval()
batch_converter = alphabet.get_batch_converter()


# 1AAY (PDB ID)

prot = "MERPYACPVESCDRRFSRSDELTRHIRIHTGQKPFQCRICMRNFSRSDHLTTHIRTHTGEKPFACDICGRKFARSDERKRHTKIHLRQKD"

dna = "TACGCCCACGC" 

# DNA embedding
inputs = dna_tokenizer.batch_encode_plus([dna], return_tensors="pt").to(device)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
with torch.no_grad():
    outputs = dna_model(           
        input_ids,
        attention_mask=attention_mask,
        encoder_attention_mask=attention_mask,
        output_hidden_states=True)
    hidden_states = outputs['hidden_states'][-1]
    dna_emb = torch.mean(hidden_states, dim=1).squeeze(0)


# Protein embedding
_, _, batch_tokens = batch_converter([("protein", prot)])
batch_tokens = batch_tokens.to(device)
with torch.no_grad():
    results = protein_model(batch_tokens, repr_layers=[33])
    max_embedding, _ = results["representations"][33][:, 1:-1, :].max(dim=1)
    protein_emb = max_embedding.squeeze(0)


score = scoring_model(protein_emb, dna_emb)
print(score[0].item())