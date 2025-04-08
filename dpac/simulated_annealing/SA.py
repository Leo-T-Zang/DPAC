import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import esm


class SimulatedAnnealing(nn.Module):
    """
    FSA(Fast Simulated Annealing)

    Parameters
    ----------------
    prot_0 : str
        initial solution, protein sequence
    na: str
        nucleotide sequence
    oracle : nn.Module
        scoring model
    prot_encoder : nn.Module
        protein encoder (ESM-2)
    prot_tokenizer : 
        ESM-2 tokenizer
    na_encoder : nn.Module
        nucleotide encoder (Nucleotide Transformer)
    na_tokenizer :
        Nucleotide Transformer tokenizer
    
    T_max :float
        initial temperature
    max_iter : int
        max iteration

    """

    def __init__(
        self, prot_0, na, oracle, 
        prot_encoder, prot_tokenizer, 
        na_encoder, na_tokenizer, device,
        constrain = None,
        mutate_region = None,
        T_max = 50, 
        max_iter = 500, 
        early_stop=150):

        super(SimulatedAnnealing, self).__init__()

        assert T_max > 0, 'T_max > 0'

        self.T_max = T_max  # initial temperature
        self.max_iter = max_iter  # max iteration
        self.early_stop = early_stop

        self.constrain = constrain

        self.prot_0 = prot_0
        self.na = na

        self.best_prot = prot_0
        self.best_score = -100

        self.T = T_max
        self.iter_cycle = 0

        self.prot_encoder = prot_encoder.to(device)
        self.prot_tokenizer = prot_tokenizer
        self.batch_converter = prot_tokenizer.get_batch_converter()

        self.na_encoder = na_encoder.to(device)
        self.na_tokenizer = na_tokenizer

        self.oracle = oracle # DPAC model

        self.device = device

        self.mutate_region = mutate_region

    def mutate(
        self,
        prot_seq: str,
        i: int,
        top_k: int = 3,
    ) -> str:

        # Prepare the protein sequence for embedding
        _, _, batch_tokens = self.batch_converter([("protein", prot_seq)])
        batch_tokens = batch_tokens.to(self.device)
        
        # Generate the protein embedding
        with torch.no_grad():
            results = self.prot_encoder(batch_tokens, repr_layers=[33])
            prot_embedding = results["representations"][33]
        
        # Get the alphabet tokens
        aa_toks = self.prot_tokenizer.all_toks
        aa_idxs = [self.prot_tokenizer.get_idx(aa) for aa in aa_toks]
        
        # Generate logits for each amino acid in the sequence
        with torch.no_grad():
            aa_logits = self.prot_encoder.lm_head(prot_embedding[0, 1:-1])[:, aa_idxs]
        
        # Replace the prediction at index i with a random sample from top-k
        topk_values, topk_indices = torch.topk(aa_logits[i], top_k)
        random_choice_from_topk = random.choice(topk_indices.tolist())
        
        prot_seq = list(prot_seq)
        prot_seq[i] = aa_toks[random_choice_from_topk]
        mutated_seq = ''.join(prot_seq)
        
        return mutated_seq

    def cool_down(self, step):
        self.T = self.T_max / float(step + 1)

    def compute_embeddings(self, prot_seq, na_seq):
            
        # DNA embedding
        inputs = self.na_tokenizer.batch_encode_plus([na_seq], return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.na_encoder(            
                input_ids,
                attention_mask=attention_mask,
                encoder_attention_mask=attention_mask,
                output_hidden_states=True)
            hidden_states = outputs['hidden_states'][-1]
            dna_embedding = torch.mean(hidden_states, dim=1).squeeze(0)


        # Protein embedding
        _, _, batch_tokens = self.batch_converter([("protein", prot_seq)])
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.prot_encoder(batch_tokens, repr_layers=[33])
            max_embedding, _ = results["representations"][33][:, 1:-1, :].max(dim=1)
            protein_embedding = max_embedding.squeeze(0)

        return protein_embedding, dna_embedding

    def score(
        self, prot_seq, na_seq
    ):
        # Compute embeddings
        protein_emb, dna_emb = self.compute_embeddings(prot_seq, na_seq)

        score = self.oracle(protein_emb, dna_emb)

        return score[0].item()

    def sequence_difference(self, seq1, seq2):
        """Calculate the percentage of different characters between two sequences."""
        if len(seq1) != len(seq2):
            raise ValueError("Sequences must be of the same length to compare.")
        differences = sum(1 for a, b in zip(seq1, seq2) if a != b)
        return differences / len(seq1)

    def sample(self):
        
        stay_counter = 0

        prot_current = self.prot_0
        score_current = self.score(prot_current, self.na)
        #print(prot_current)
        print(f"Initial score: {score_current}")

        if self.mutate_region:
            # Build a list of indices to mutate
            mutate_region = self.mutate_region.copy()

        log_data = {'Step': [], 'Sequence': [], 'Score': []}

        for i in range(1, self.max_iter + 1):
            # print('step', i)
            # randomly sample number of mutated index
            if self.mutate_region:
                mut_idx = random.choice(mutate_region)
                print(f"Mutating index {mut_idx}")
            else:
                mut_idx = np.random.randint(0, len(prot_current))

            prot_new = self.mutate(prot_current, mut_idx, top_k=3)

            score_new = self.score(prot_new, self.na)

            # Metropolis
            df = (1 - score_new) - (1 - score_current)

            if df < 0 or np.exp(-df / self.T) > np.random.rand():
                
                prot_current = prot_new
                score_current = score_new

                # Remove the mutated index from the list if accepted
                if self.mutate_region:
                    mutate_region.remove(mut_idx)

            self.cool_down(i)

            log_data['Step'].append(i)
            log_data['Sequence'].append(prot_current)
            log_data['Score'].append(score_current)

            # print('current score', score_current)
            # print('best score', self.best_score)
            
            if score_current > self.best_score:
                self.best_prot = prot_current
                self.best_score = score_current
                stay_counter = 0
            
            else:
            # Early stop if the best score doesn't change
                stay_counter += 1
                # print(f"Stay counter: {stay_counter}")
                if stay_counter >= self.early_stop:
                    # print(f"Early stop at iteration {i}")
                    break

            
            if self.constrain is not None:
                difference = self.sequence_difference(prot_current, self.prot_0)
                print(f"Mutation constrained at step {i}: {difference:.2f}")
                if difference >= self.constrain:
                    break

            if self.mutate_region and len(mutate_region) == 0:
                break

        # Convert log data to DataFrame
        log_df = pd.DataFrame(log_data)

        return self.best_prot, self.best_score, log_df




