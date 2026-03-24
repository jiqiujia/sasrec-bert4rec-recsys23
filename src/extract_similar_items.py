"""
Extract item embeddings from a SASRec checkpoint and compute top-k similar items
for each item based on cosine similarity.

Usage:
    python extract_similar_items.py --ckpt_path <path_to_ckpt> --top_k 10 --output_path similar_items.csv
"""

import argparse
import torch
import numpy as np
import pandas as pd


def load_embedding_from_checkpoint(ckpt_path):
    """Load item embedding weights from a PyTorch Lightning checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    # Print all keys for reference
    print("Keys in checkpoint state_dict:")
    for key in state_dict.keys():
        print(f"  {key}: {state_dict[key].shape}")

    # SASRec: model.item_emb.weight
    # BERT4Rec / RNN / GPT4Rec: model.embed_layer.weight
    embedding_key = None
    for key in state_dict:
        if "item_emb.weight" in key or "embed_layer.weight" in key:
            embedding_key = key
            break

    if embedding_key is None:
        raise KeyError("Cannot find embedding layer in checkpoint. "
                       f"Available keys: {list(state_dict.keys())}")

    embedding_weight = state_dict[embedding_key]
    print(f"\nUsing embedding key: '{embedding_key}'")
    print(f"Embedding shape: {embedding_weight.shape}  "
          f"(num_items_with_padding x hidden_dim)")

    return embedding_weight


def compute_topk_similar(embedding_weight, top_k=10, batch_size=1024):
    """
    Compute top-k most similar items for every item using cosine similarity.

    Args:
        embedding_weight: Tensor of shape (num_items, hidden_dim).
                          Index 0 is the padding embedding and will be skipped.
        top_k: Number of similar items to retrieve per item.
        batch_size: Batch size for computing similarity to avoid OOM.

    Returns:
        sim_item_ids: np.ndarray of shape (num_real_items, top_k) — similar item IDs
        sim_scores:   np.ndarray of shape (num_real_items, top_k) — similarity scores
        item_ids:     np.ndarray of shape (num_real_items,) — the source item IDs
    """
    # Skip padding index 0
    embeddings = embedding_weight[1:]  # (num_real_items, hidden_dim)
    num_items = embeddings.shape[0]
    item_ids = np.arange(1, num_items + 1)  # real item IDs start from 1

    # Normalize embeddings for cosine similarity
    norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    embeddings_normed = embeddings / norms

    all_topk_ids = []
    all_topk_scores = []

    print(f"\nComputing top-{top_k} similar items for {num_items} items ...")

    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        # (batch, hidden_dim) @ (hidden_dim, num_items) -> (batch, num_items)
        sim_matrix = embeddings_normed[start:end] @ embeddings_normed.T

        # Mask self-similarity by setting diagonal elements to -inf
        for i in range(end - start):
            sim_matrix[i, start + i] = -float("inf")

        # Get top-k
        scores, indices = torch.topk(sim_matrix, k=top_k, dim=1)

        # Convert indices back to real item IDs (offset by 1 since we skipped padding)
        real_ids = indices.numpy() + 1

        all_topk_ids.append(real_ids)
        all_topk_scores.append(scores.numpy())

        if (start // batch_size) % 10 == 0:
            print(f"  Processed {end}/{num_items} items")

    sim_item_ids = np.concatenate(all_topk_ids, axis=0)
    sim_scores = np.concatenate(all_topk_scores, axis=0)

    return sim_item_ids, sim_scores, item_ids


def save_results(item_ids, sim_item_ids, sim_scores, output_path, top_k):
    """Save the top-k similar items to a CSV file."""
    rows = []
    for i, src_id in enumerate(item_ids):
        for rank in range(top_k):
            rows.append({
                "item_id": int(src_id),
                "rank": rank + 1,
                "similar_item_id": int(sim_item_ids[i, rank]),
                "cosine_similarity": float(f"{sim_scores[i, rank]:.6f}")
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(f"Total rows: {len(df)}  ({len(item_ids)} items x {top_k} neighbors)")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract item embeddings and compute top-k similar items.")
    parser.add_argument("--ckpt_path", type=str,
                        default="lightning_logs/version_0/checkpoints/epoch=5-step=1890.ckpt",
                        help="Path to the PyTorch Lightning checkpoint file.")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of most similar items to retrieve per item.")
    parser.add_argument("--output_path", type=str, default="similar_items.csv",
                        help="Output CSV file path.")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for similarity computation.")
    args = parser.parse_args()

    # 1. Load embedding
    embedding_weight = load_embedding_from_checkpoint(args.ckpt_path)

    # 2. Compute top-k similar items
    sim_item_ids, sim_scores, item_ids = compute_topk_similar(
        embedding_weight, top_k=args.top_k, batch_size=args.batch_size)

    # 3. Save results
    df = save_results(item_ids, sim_item_ids, sim_scores, args.output_path, args.top_k)

    # 4. Print some examples
    print("\n--- Sample results (first 5 items) ---")
    for item_id in item_ids[:5]:
        subset = df[df["item_id"] == item_id]
        neighbors = list(zip(subset["similar_item_id"], subset["cosine_similarity"]))
        print(f"Item {item_id}: {neighbors}")


if __name__ == "__main__":
    main()
