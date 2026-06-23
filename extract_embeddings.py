import torch
import numpy as np
from evo2 import Evo2

MODEL_ID = "arcinstitute/evo2_7b" 
LAYER_NAME = 'blocks.28.mlp.l3'
PAD_ID = 0
BATCH_SIZE = 1
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu" 
          )
AVERAGE = False

print(f"Loading {MODEL_ID} ...")

model = Evo2("evo2_7b")


print("Model loaded.\n")

# Embedding extraction

def extract_embeddings(
        sequences: list[str],
    model,
    layer: str = LAYER_NAME,
    batch_size: int = BATCH_SIZE,
    pad_id: int = PAD_ID,
    average: bool = AVERAGE
) -> np.ndarray:


      """
    Returns mean-pooled hidden-state embeddings, shape (N, hidden_dim).
 
    Args:
        sequences:  list of DNA strings (A/T/C/G, upper-case recommended)
        layer:      hidden layer nsmr to pool
        batch_size: sequences per forward pass
        average: if average embeddings across sequence length
    """
      all_embeddings = []
      for start in range(0, len(sequences), batch_size):
        seqs = sequences[start:start + batch_size]
        token_ids = [model.tokenizer.tokenize(seq) for seq in seqs]
        lengths = [len(t) for t in token_ids]
        max_length = max(lengths)
        padded = [t + [pad_id] * (max_length - len(t)) for t in token_ids]
        input_ids = torch.tensor(padded, dtype = torch.int).to(DEVICE)

        with torch.no_grad():
            _, embeddings = model.forward(input_ids,
                                            return_embeddings = True,
                                            layer_names = [layer])
            
            hidden = embeddings[layer] # B, L, D

            if average:
                    mask = torch.zeros((hidden.shape[0], hidden.shape[1]), dtype = torch.bool, device = DEVICE)
                    
                    for i,l in enumerate(lengths):
                        mask[i,:l] = True
                    
                    # Average across sequence length - will remove it later!
                    mask_expanded = mask.unsqueeze(-1).float() # B, L, 1
                    pooled = (hidden * mask_expanded).sum(1)
                    pooled = pooled/mask_expanded.sum()
            else:
                pooled = hidden
            all_embeddings.append(pooled.cpu().numpy())
            print(f"Processed {min(start + batch_size, len(sequences))}/{len(sequences)} sequences")

      return np.concatenate(all_embeddings, axis = 0)

if __name__ == "__main__":
    seqs = np.load(f"output/NPC1_dna_seq_avg_{AVERAGE}.npy")
    embeddings = extract_embeddings(seqs, model, LAYER_NAME, BATCH_SIZE, AVERAGE)
    np.save("embeddings.npy", embeddings)

                

                


                  
            