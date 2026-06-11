"""Rebuild the `artworks` reference table in pgvector from embeddings.npz.

The original one-off loader was never committed; this reproduces it. It reads the SAME
embeddings.npz that trains the classifier and emits a CSV for psql `\\copy`, so it needs
no Python DB driver. Run it after re-extracting embeddings with a new CLIP model (e.g. the
336px upgrade), then load the CSV (see the retrain runbook).

By default it only PREVIEWS (dry run). Pass --out artworks.csv to write the file.

    python load_artworks.py                      # dry run: preview first rows
    python load_artworks.py --out artworks.csv   # write the CSV

The similarity search depends only on the embedding vector; artist/title are best-effort
cosmetic fields parsed from the WikiArt filename. style and the MinIO key are exact.
"""
import os
import sys
import csv
import argparse
import numpy as np

INPUT_FILE = "embeddings.npz"


def parse_artist_title(filename):
    """Best-effort split of a WikiArt filename into (artist, title)."""
    name = os.path.splitext(filename)[0]
    if "_" in name:
        artist_part, title_part = name.split("_", 1)
    else:
        artist_part, title_part = name, name
    artist = artist_part.replace("-", " ").strip().title() or "Unknown"
    title = title_part.replace("-", " ").strip().title() or name
    return artist, title


def rows_from_npz(npz_path):
    data = np.load(npz_path)
    embeddings = data["embeddings"]
    labels = data["labels"]
    paths = data["paths"]
    for emb, style, path in zip(embeddings, labels, paths):
        filename = os.path.basename(str(path).replace("\\", "/"))
        style = str(style)
        # MinIO key under the `dataset` bucket (mc mirror preserved <Style>/<file>).
        # The Java presigner treats a key without "/dataset/" as the object key directly.
        key = f"{style}/{filename}"
        artist, title = parse_artist_title(filename)
        vec = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
        yield artist, title, style, key, vec


def main():
    ap = argparse.ArgumentParser(description="Rebuild artworks table CSV from embeddings.npz")
    ap.add_argument("--input", default=INPUT_FILE, help="embeddings.npz path")
    ap.add_argument("--out", help="write CSV here (omit for a dry-run preview)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found. Run extract_features.py first.")
        sys.exit(1)

    if not args.out:
        print("DRY RUN - first 5 rows (pass --out artworks.csv to write):\n")
        for i, (artist, title, style, key, vec) in enumerate(rows_from_npz(args.input)):
            if i >= 5:
                break
            print(f"  artist={artist!r} title={title!r} style={style!r}")
            print(f"  image_s3_url={key!r}  embedding={vec[:40]}...\n")
        print("OK (dry run). Review the parsed fields, then re-run with --out.")
        return

    n = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["artist", "title", "style", "image_s3_url", "embedding"])
        for row in rows_from_npz(args.input):
            w.writerow(row)
            n += 1
    print(f"Wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
