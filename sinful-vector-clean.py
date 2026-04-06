"""
PCA-based de-noising of sin vectors by projecting out neutral components.

(c) Aritro 'sortira' Shome
"""
from loguru import logger
import torch
import os
from pathlib import Path
from sklearn.decomposition import PCA
import numpy as np

logger.remove()
logger.add(lambda msg: print(msg, end=""), colorize=True, format="<level>{message}</level>")

INPUT_PATH = "data/raw-sinful-vectors"
OUTPUT_PATH = "data/sinful-vectors-clean"
SINS = ["pride", "envy", "gluttony", "sloth", "wrath", "lust", "greed"]
N_COMPONENTS = 7

Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

logger.info(f"Loading neutral vectors from {INPUT_PATH}...")
neutral_vector = torch.load(os.path.join(INPUT_PATH, "neutral.pt"))
logger.info(f"Neutral shape (raw): {neutral_vector.shape}")

neutral_vector = neutral_vector.squeeze()
logger.info(f"Neutral shape (squeezed): {neutral_vector.shape}")

num_layers, d_model = neutral_vector.shape

logger.info(f"Performing PCA ({N_COMPONENTS} components) on neutral vectors...")
neutral_flat = neutral_vector.reshape(-1, d_model).cpu().numpy()
pca = PCA(n_components=N_COMPONENTS)
pca.fit(neutral_flat)

W_neutral = torch.from_numpy(pca.components_).float()
logger.success(f"PCA complete. Found {N_COMPONENTS} principal components.")

logger.info("De-noising sin vectors...")
for sin in SINS:
    sin_vector = torch.load(os.path.join(INPUT_PATH, f"{sin}.pt")).squeeze()
    clean_vector = sin_vector.clone()
    
    for layer in range(num_layers):
        v_raw = sin_vector[layer]
        
        for j in range(N_COMPONENTS):
            w_j = W_neutral[j]
            projection = torch.dot(v_raw, w_j)
            clean_vector[layer] = clean_vector[layer] - projection * w_j
    
    output_file = os.path.join(OUTPUT_PATH, f"{sin}.pt")
    torch.save(clean_vector, output_file)
    logger.info(f"Saved {sin} {clean_vector.shape} -> {output_file}")

torch.save(W_neutral, os.path.join(OUTPUT_PATH, "pca_components.pt"))
torch.save(neutral_vector, os.path.join(OUTPUT_PATH, "neutral_vector.pt"))
logger.success(f"All clean vectors saved to {OUTPUT_PATH}")
