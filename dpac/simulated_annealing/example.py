import os
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
import esm
import pandas as pd


from ..model import DPAC
from .SA import SimulatedAnnealing
from ..utils import ModelConfig, set_seed


set_seed(42)

def mutate_protein(sequence, mutation_percent):
    '''
    Mutate a protein sequence by randomly changing a percentage of its amino acids.
    '''
    length = len(sequence)
    num_mutations = int(length * (mutation_percent / 100))
    indices = random.sample(range(length), num_mutations)
    
    # Convert the sequence to a list to mutate it
    sequence_list = list(sequence)
    
    # All possible amino acids
    amino_acids = 'ARDCQEGHILKMNFPSTWYV'
    
    for index in indices:
        original_amino_acid = sequence_list[index]
        # Create a list of possible mutations excluding the original amino acid
        possible_mutations = [aa for aa in amino_acids if aa != original_amino_acid]
        
        # Randomly select a new amino acid from the possible mutations
        sequence_list[index] = random.choice(possible_mutations)
    
    # Convert list back to string
    mutated_sequence = ''.join(sequence_list)
    
    return mutated_sequence, indices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config = ModelConfig()
scoring_model = DPAC(model_config).to(device)
dpac_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
weight_path = os.path.join(dpac_dir, 'weight.pt')
scoring_model.load_state_dict(torch.load(weight_path)['model_state_dict'])
scoring_model.eval()

dna_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)
dna_model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True).to(device)
dna_model.eval()

protein_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
protein_model = protein_model.to(device)
protein_model.eval()
batch_converter = alphabet.get_batch_converter()

# 1AAY
# Randomly mutate 30% of AA and use SA to optimize the protein sequence

prot_0 = "MERPYACPVESCDRRFSRSDELTRHIRIHTGQKPFQCRICMRNFSRSDHLTTHIRTHTGEKPFACDICGRKFARSDERKRHTKIHLRQKD"

na = "TACGCCCACGC" 

constraint = None
# Initialize the Simulated Annealing model

mutation_percent = 30 # percent

mutated_sequence, mutation_indices = mutate_protein(prot_0, mutation_percent)
print("Mutated Sequence:", mutated_sequence)
print("Mutation Indices:", mutation_indices)

prot_0 = mutated_sequence

sa = SimulatedAnnealing(prot_0, na, scoring_model, protein_model, 
                        alphabet, dna_model, dna_tokenizer, device, 
                        constrain=constraint, mutate_region=None)
with torch.no_grad():
    # Run the Simulated Annealing algorithm
    best_prot, best_score, log_df = sa.sample()
print(f"Best protein sequence: {best_prot}")
print(f"Best score: {best_score}")
log_df.append({'Best Protein': best_prot, 'Best Score': best_score}, ignore_index=True)
print(log_df)
#log_df.to_csv(f'./1AAY/log_{constraint}_{mutation_percent}.csv', index=False)