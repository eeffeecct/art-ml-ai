"""Rebuild embeddings.npz from vectors already stored in the artworks table.

Lets you retrain the classifier WITHOUT the raw images / CLIP — the embeddings that
trained it already live in pgvector. First export them to CSV (one line, see runbook):

    \\copy (SELECT style, embedding FROM artworks) TO 'train.csv' WITH (FORMAT csv)

then:

    python npz_from_db.py train.csv      # -> embeddings.npz next to train.csv

The .npz is written beside the input CSV, so run train_classifier.py from that folder.
"""
import os
import sys
import csv
import numpy as np


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "train.csv"
    if not os.path.exists(src):
        print(f"Error: {src} not found. Export it from the artworks table first.")
        sys.exit(1)

    out = os.path.join(os.path.dirname(os.path.abspath(src)), "embeddings.npz")
    embs, labels = [], []
    with open(src, newline="", encoding="utf-8-sig") as f:  # utf-8-sig strips a BOM if present
        for row in csv.reader(f):
            if len(row) != 2:
                continue
            style, vec = row
            embs.append(np.array(vec.strip().strip("[]").split(","), dtype=np.float32))
            labels.append(style)

    if not embs:
        print("Error: no rows parsed. Expected CSV columns: style,embedding")
        sys.exit(1)

    X = np.vstack(embs).astype(np.float32)
    np.savez_compressed(out, embeddings=X, labels=np.array(labels), paths=np.array(labels))
    print(f"Wrote {out}: {X.shape[0]} rows, dim {X.shape[1]}, {len(set(labels))} styles")


if __name__ == "__main__":
    main()
