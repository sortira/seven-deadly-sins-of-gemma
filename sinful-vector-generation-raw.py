"""
Seven deadly sins vector extraction and raw vector generation.

(c) Aritro 'sortira' Shome
"""
from loguru import logger
from transformer_lens import HookedTransformer
import torch
import os
from pathlib import Path

logger.remove()
logger.add(lambda msg: print(msg, end=""), colorize=True, format="<level>{message}</level>")

BASE_DATA_PATH = "data"
OUTPUT_PATH = "data/raw-sinful-vectors"
SINS = ["pride", "envy", "gluttony", "sloth", "wrath", "lust", "greed", "neutral"]
TOKEN_START = 50

Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

logger.info("Loading Gemma 2 2B model...")
model = HookedTransformer.from_pretrained("google/gemma-2-2b")
num_layers = model.cfg.n_layers

logger.info(f"Model has {num_layers} layers")
logger.info(f"Processing {len(SINS)} sins across {num_layers} layers...")

emotion_vectors = {sin: {} for sin in SINS}

for layer in range(num_layers):
    logger.info(f"Processing layer {layer}...")
    
    for sin in SINS:
        folder = os.path.join(BASE_DATA_PATH, sin)
        story_files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
        
        temporal_averages = []
        
        for story_file in story_files:
            story_path = os.path.join(folder, story_file)
            with open(story_path, "r", encoding="utf-8") as f:
                story_text = f.read().strip()
            
            logits, cache = model.run_with_cache(story_text)
            resid = cache[f"blocks.{layer}.hook_resid_pre"]
            
            seq_len = resid.shape[1]
            if seq_len > TOKEN_START:
                temporal_avg = resid[:, TOKEN_START:, :].mean(dim=1)
                temporal_averages.append(temporal_avg)
            else:
                logger.warning(f"{sin}/{story_file} has only {seq_len} tokens, skipping")
        
        if temporal_averages:
            prototype = torch.stack(temporal_averages).mean(dim=0)
            emotion_vectors[sin][layer] = prototype

global_means = {}
for layer in range(num_layers):
    layer_prototypes = torch.stack([emotion_vectors[sin][layer] for sin in SINS])
    global_means[layer] = layer_prototypes.mean(dim=0)

logger.info("Computing raw vectors and saving...")

for sin in SINS:
    raw_vectors_layers = []
    for layer in range(num_layers):
        raw_vector = emotion_vectors[sin][layer] - global_means[layer]
        raw_vectors_layers.append(raw_vector)
    
    raw_vectors_stacked = torch.stack(raw_vectors_layers).squeeze()
    
    output_file = os.path.join(OUTPUT_PATH, f"{sin}.pt")
    torch.save(raw_vectors_stacked, output_file)
    logger.info(f"Saved {sin} vectors {raw_vectors_stacked.shape} -> {output_file}")

global_means_dict = {layer: global_means[layer].squeeze() for layer in range(num_layers)}
torch.save(global_means_dict, os.path.join(OUTPUT_PATH, "global_means.pt"))
logger.success(f"Completed. All vectors saved to {OUTPUT_PATH}") 
